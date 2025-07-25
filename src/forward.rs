

use std::fs::{File, read};
use anyhow::{Error as E, Result};
use candle_core::{Tensor, Device, safetensors::load_buffer};
use candle_nn::VarBuilder;
use candle_nn::ops::sigmoid;
use candle_core::DType;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::Tokenizer;
use ndarray::{Array1, Array2, Array3, array};
use ndarray_npy::NpzReader;
use serde_json;
use crate::interpolate::trilinear_interpolation;

pub fn load_constants() -> Result<(
    BertModel, Tokenizer,
    Array1<bool>, Array2<f32>, Array2<f32>,          // arrays for attention mask surface plotting
    Tensor, Tensor, Tensor, Tensor, Tensor,          // aligner and decoder weights
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
), Box<dyn std::error::Error>> {

    // Load
    let device = Device::Cpu;
    let buffer = read("static/models/model.safetensors").expect("failed to load model.safetensors");
    let vb = VarBuilder::from_buffered_safetensors(buffer, DTYPE, &device)
        .expect("Loading VarBuilder failed.");
    let config: Config = serde_json::from_reader(File::open("static/models/config.json")?)?;

    let tokenizer = Tokenizer::from_file("static/models/tokenizer.json");
    let title_embeddings = {
        let data = read("static/models/titles.safetensors")?;
        load_buffer(&data, &device)?
            .get("titles")
            .ok_or("Tensor 'titles' not found in safetensors file")?
            .clone()
            .to_dtype(DType::F32)?
    };

    let model = BertModel::load(vb, &config).expect("BertModel failed.");
    let mut npz = NpzReader::new(File::open("static/pkg/constants.npz")?)?;
    let mask : Array1<bool> = npz.by_name("MASK")?;
    let r_reg_fus : Array2<f32> = npz.by_name("R_REG_FUS")?;
    let l_reg_fus : Array2<f32> = npz.by_name("L_REG_FUS")?;

    // Projections
    //   unpack aligner
    let device = Device::Cpu;
    let aligner_data = read("static/models/aligner.safetensors").unwrap();
    let aligner_tensors = load_buffer(&aligner_data, &device)?;
    let aligner_w0 = aligner_tensors
        .get("align_w0").ok_or("align_w0 tensor not found")? // load
        .to_dtype(DType::F32)?.t().unwrap();                 // cast, transpose, unwrap
    let aligner_b0 = aligner_tensors
        .get("align_b0").ok_or("align_b0 tensor not found")?
        .to_dtype(DType::F32)?.unsqueeze(0).unwrap();
    let aligner_w1 = aligner_tensors
        .get("align_w1").ok_or("align_w1 tensor not found")?
        .to_dtype(DType::F32)?.t().unwrap();
    let aligner_b1 = aligner_tensors
        .get("align_b1").ok_or("align_b1 tensor not found")?
        .to_dtype(DType::F32)?.unsqueeze(0).unwrap();

    //   unpack decoder
    let decoder_data = read("static/models/decoder.safetensors")?;
    let decoder_tensors = load_buffer(&decoder_data, &device)?;
    let decoder_w0 = decoder_tensors
        .get("decode_w0").ok_or("decoder_w0 tensor not found")?
        .to_dtype(DType::F32)?.t().unwrap();
    let decoder_b0 = decoder_tensors
        .get("decode_b0").ok_or("decoder_b0 tensor not found")?
        .to_dtype(DType::F32)?.unsqueeze(0).unwrap();
    let decoder_w1 = decoder_tensors
        .get("decode_w1").ok_or("decoder_w1 tensor not found")?
        .to_dtype(DType::F32)?.t().unwrap();
    let decoder_b1 = decoder_tensors
        .get("decode_b1").ok_or("decoder_b1 tensor not found")?
        .to_dtype(DType::F32)?.unsqueeze(0).unwrap();
    let decoder_w2 = decoder_tensors
        .get("decode_w2").ok_or("decoder_w2 tensor not found")?
        .to_dtype(DType::F32)?.t().unwrap();
    let decoder_b2 = decoder_tensors
        .get("decode_b2").ok_or("decoder_b2 tensor not found")?
        .to_dtype(DType::F32)?.unsqueeze(0).unwrap();

    Ok((
        // Transformer
        model,
        tokenizer.unwrap(),
        mask,
        // For surface plotting
        l_reg_fus,
        r_reg_fus,
        // Precompute embeddings
        title_embeddings,
        aligner_w0,
        aligner_b0,
        aligner_w1,
        aligner_b1,
        decoder_w0,
        decoder_b0,
        decoder_w1,
        decoder_b1,
        decoder_w2,
        decoder_b2
    ))
}

pub fn text_query(
    query: &str,
    model: &BertModel,
    tokenizer: &Tokenizer,
    title_embeddings: &Tensor,
    mask: &Array1<bool>,
    l_reg_fus: &Array2<f32>,
    r_reg_fus: &Array2<f32>,
    aligner_w0: &Tensor,
    aligner_b0: &Tensor,
    aligner_w1: &Tensor,
    aligner_b1: &Tensor,
    decoder_w0: &Tensor,
    decoder_b0: &Tensor,
    decoder_w1: &Tensor,
    decoder_b1: &Tensor,
    decoder_w2: &Tensor,
    decoder_b2: &Tensor
) -> Result<(Vec<i32>, Vec<f32>, Array3<f32>), Box<dyn std::error::Error>>{

    // Compute embedding for the query
    let sentences = [query];

    let tokens = tokenizer
        .encode_batch(sentences.to_vec(), true)
        .map_err(E::msg).unwrap();

    let device = &model.device;

    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), device).unwrap())
        })
        .collect::<Result<Vec<_>>>().expect("Collecting tokens failed.");

    let token_ids = Tensor::stack(&token_ids, 0)
        .expect("Stacking tokens failed.");
    let token_type_ids = token_ids.zeros_like()
        .expect("Zero array failed.");

    let attention_mask = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_attention_mask().to_vec();
            Ok(Tensor::new(tokens.as_slice(), device)?)
        }).collect::<Result<Vec<_>>>().unwrap();
    let attention_mask = Tensor::stack(&attention_mask, 0)
        .expect("Attention mask failed.");

    let embeddings = model.forward(&token_ids, &token_type_ids, Some(&attention_mask))
        .expect("Embeddings failed.");

    let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3().unwrap();
    let embeddings = (embeddings.sum(1).unwrap() / (n_tokens as f64)).unwrap();

    let query_embedding_1d = embeddings.get(0).expect("Failed to unpack embeddings");
    let query_embedding_2d = query_embedding_1d.unsqueeze(0).unwrap();
    let n_titles = title_embeddings.dim(0).expect("Failed to get number of tiles");

    // Cosine similarity
    let mut similarities = vec![];
    for i in 0..n_titles {
        let e_i = title_embeddings.get(i).unwrap();
        let sum_ij = (&query_embedding_1d * &e_i)
            .unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        let sum_i2 = (&query_embedding_1d * &query_embedding_1d)
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
    //   aligner
    let x0 = (
        query_embedding_2d.broadcast_matmul(&aligner_w0).unwrap() + aligner_b0
    ).unwrap().relu().unwrap();

    let neuro_aligned = (
        x0.broadcast_matmul(&aligner_w1) + aligner_b1
    ).unwrap();

    // decoder
    let x0 = (
        neuro_aligned.matmul(&decoder_w0).unwrap() + decoder_b0
    ).unwrap().relu().unwrap();

    let x1 = (
        x0.matmul(&decoder_w1).unwrap() + decoder_b1
    ).unwrap().relu().unwrap();

    let x2 = (
        x1.matmul(&decoder_w2).unwrap() + decoder_b2
    ).unwrap();

    let neuro_pred = sigmoid(&x2).unwrap().reshape((28542,)).unwrap();

    // Place prediction onto volume
    let mut img3d: Array3<f32> = Array3::zeros((46, 55, 46));
    let mut img3d_swapped: Array3<f32> = Array3::zeros((46, 55, 46));
    let mut ii : usize = 0;
    let mut jj : usize = 0;
    for i in 0..46{
        for j in 0..55{
            for k in 0..46{
                if mask[jj]{
                    let val =  neuro_pred.get(ii).unwrap().to_scalar::<f32>().unwrap();
                    img3d[[i, j, k]] = val;
                    img3d_swapped[[k, j, i]] = val;
                    ii += 1;
                }
                jj += 1;
            }
        }
    }
    // let img3d_swapped = img3d.permuted_axes([2, 1, 0]).to_owned();

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
