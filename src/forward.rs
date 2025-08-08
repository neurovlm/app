// Forward pass through models
use std::fs::{File, read};
use anyhow::Result;
use candle_core::{Tensor as CandleTensor, Device, safetensors::load_buffer, Shape};
use candle_nn::ops::sigmoid;
use tokenizers::Tokenizer;
use ndarray::{Array1, Array2, Array3, array};
use ndarray_npy::NpzReader;
use tch::{Tensor, CModule};
use tch::nn::Module;
use crate::interpolate::trilinear_interpolation;

fn tch_to_candle(t: &Tensor) -> anyhow::Result<CandleTensor> {
    // Convert shape from Vec<i64> to Vec<usize>
    let shape: Vec<usize> = t.size().into_iter().map(|x| x as usize).collect();

    // Extract data (requires owned Tensor, hence .shallow_clone())
    let data: Vec<f32> = Vec::<f32>::try_from(t)?;

    // Create Candle tensor
    let candle = CandleTensor::from_vec(data, Shape::from(shape), &Device::Cpu)?;
    Ok(candle)
}

// Model structures
pub struct Specter2 {
    model: CModule,
    tokenizer: Tokenizer
}

impl Specter2 {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        Ok(Self {
            model: CModule::load(model_path)?,
            tokenizer: Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!("{}", e))?
        })
    }

    pub fn encode(&self, text: &str) -> Result<Tensor> {
        let encoding = self.tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!("{}", e))?;
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        let input_ids = Tensor::from_slice(&input_ids).unsqueeze(0);
        let attention_mask = Tensor::from_slice(&attention_mask).unsqueeze(0);
        Ok(self.model.forward_ts(&[input_ids, attention_mask])?)
    }
}

pub struct GenericModel {
    model: CModule
}
impl GenericModel {
    pub fn new(model_path: &str) -> Result<Self>{
        Ok(Self {
            model: CModule::load(model_path)?
        })
    }
    pub fn forward(&self, latent: &Tensor) -> Result<Tensor>{
        Ok(self.model.forward(latent))
    }
}

pub fn load_constants() -> Result<(
    // text_encoder, neuro_decoder, aligner, mask, l_reg_fus, r_reg_fus, title_embeddings
    Specter2, GenericModel, GenericModel, Array1<bool>, Array2<f32>, Array2<f32>, CandleTensor
), Box<dyn std::error::Error>> {
    // models and embeddings
    let text_encoder = Specter2::new(
        "/home/rphammonds/neurovlm/static/models/specter2_traced.pt",
        "/home/rphammonds/neurovlm/static/models/tokenizer/tokenizer.json"
    )?;
    let neuro_decoder = GenericModel::new("/home/rphammonds/neurovlm/static/models/decoder_traced.pt")?;
    let aligner = GenericModel::new("/home/rphammonds/neurovlm/static/models/aligner_traced.pt")?;

    let title_embeddings = {
        let data = read("/home/rphammonds/neurovlm/static/models/latent_text_specter2_adhoc_query.safetensors")?;
        load_buffer(&data, &Device::Cpu)?.remove("latent").ok_or("could not load tensor")?
    };

    // for plotting
    let mut npz = NpzReader::new(File::open("/home/rphammonds/neurovlm/static/pkg/constants.npz")?)?;
    let mask : Array1<bool> = npz.by_name("MASK")?;
    let r_reg_fus : Array2<f32> = npz.by_name("R_REG_FUS")?;
    let l_reg_fus : Array2<f32> = npz.by_name("L_REG_FUS")?;

    Ok((
        // Models
        text_encoder,
        neuro_decoder,
        aligner,
        // For surface plotting
        mask,
        l_reg_fus,
        r_reg_fus,
        // Precompute embeddings
        title_embeddings,
    ))
}

pub fn text_query(
    query: &str,
    specter: &Specter2,
    neuro_decoder: &GenericModel,
    aligner: &GenericModel,
    title_embeddings: &CandleTensor,
    mask: &Array1<bool>,
    l_reg_fus: &Array2<f32>,
    r_reg_fus: &Array2<f32>,
) -> Result<(Vec<i32>, Vec<f32>, Array3<f32>), Box<dyn std::error::Error>>{

    // Compute embedding for the query
    let embedding_tch = specter.encode(query)?;
    let query_embedding = tch_to_candle(&embedding_tch)?;
    let n_titles = title_embeddings.dim(0)?;

    // Cosine similarity
    let mut similarities = vec![];
    for i in 0..n_titles {
        let e_i = title_embeddings.get(i).unwrap();
        let sum_ij = (&query_embedding * &e_i)
            .unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        let sum_i2 = (&query_embedding * &query_embedding)
            .unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        let sum_j2 = (&e_i * &e_i)
            .unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
        similarities.push((cosine_similarity, i))
    }
    similarities.sort_by(|u, v| v.0.total_cmp(&u.0));

    // weight titles based on similarity to query
    let top_k : usize = 100;
    let mut top_inds : Vec<i32> = vec![0; top_k];
    for i in 0..top_k{
        top_inds[i] = similarities[i].1 as i32;
    }

    let neuro_pred_tch = neuro_decoder.forward(&aligner.forward(&embedding_tch)?)?;
    let neuro_pred_unormalized = sigmoid(&tch_to_candle(&neuro_pred_tch)?)?;
    let max_val = neuro_pred_unormalized.max(0)?;
    let neuro_pred = neuro_pred_unormalized.broadcast_div(&max_val)?;

    // Place prediction onto volume
    let mut img3d: Array3<f32> = Array3::zeros((46, 55, 46));
    let mut img3d_swapped: Array3<f32> = Array3::zeros((46, 55, 46));
    let mut ii : usize = 0;
    let mut jj : usize = 0;
    for i in 0..46{
        for j in 0..55{
            for k in 0..46{
                if mask[jj]{
                    let val = neuro_pred.get(ii).unwrap().to_scalar::<f32>().unwrap();
                    img3d[[i, j, k]] = val;
                    img3d_swapped[[k, j, i]] = val;
                    ii += 1;
                }
                jj += 1;
            }
        }
    }
    let rzs : Array2<f32> = array![
        [0.25, 0.  , 0.  ],
        [0.  , 0.25, 0.  ],
        [0.  , 0.  , 0.25]
    ];
    let trans : Array1<f32> = array![22.5, 31.5, 18. ];
    let coords_l = l_reg_fus.dot(&rzs) + &trans;
    let coords_r = r_reg_fus.dot(&rzs) + &trans;
    let ix = Array1::range(0., 46., 1.0);
    let iy = Array1::range(0., 55., 1.0);
    let iz = Array1::range(0., 46., 1.0);

    let lsurface = trilinear_interpolation((&ix, &iy, &iz), img3d.view(), &coords_l);
    let rsurface = trilinear_interpolation((&ix, &iy, &iz), img3d.view(), &coords_r);

    let mut lsurface_vec: Vec<f32> = lsurface.to_vec();
    let mut rsurface_vec: Vec<f32> = rsurface.to_vec();
    lsurface_vec.append(&mut rsurface_vec);

    Ok((top_inds, lsurface_vec, img3d_swapped))
}
