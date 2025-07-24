

use std::fs::{File, read};
use anyhow::{Error as E, Result};
use candle_core::{Tensor, Device, safetensors::load_buffer, DType};
use candle_nn::VarBuilder;
use candle_nn::ops::sigmoid;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::{PaddingParams, Tokenizer};
use ndarray::{Array1, Array2, Array3, array};
use ndarray_npy::NpzReader;
use serde_json;
use crate::interpolate::trilinear_interpolation;

pub fn load_constants() -> Result<(BertModel, Tokenizer, Array1<bool>, Array2<f32>, Array2<f32>, Tensor), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    let buffer = read("static/pkg_encoder/assets/model.safetensors").expect("failed to load model.safetensors");
    let vb = VarBuilder::from_buffered_safetensors(buffer, DTYPE, &device)
        .expect("Loading VarBuilder failed.");

    //let config: Config = serde_json::from_reader(File::open("static/pkg_encoder/assets/config.json"))
    //    .expect("Loading config failed.");

    let config: Config = serde_json::from_reader(File::open("static/pkg_encoder/assets/config.json")?)?;

    let tokenizer = Tokenizer::from_file("static/pkg_encoder/assets/tokenizer.json");
    // let title_embeddings = SafeTensors::deserialize(&read("static/pkg_encoder/assets/titles.safetensors")?)?;
    let title_embeddings = {
        let data = read("static/pkg_encoder/assets/titles.safetensors")?;
        load_buffer(&data, &device)?
            .get("titles")
            .ok_or("Tensor 'titles' not found in safetensors file")?
            .clone()
            //.to_dtype(DType::F32)?
    };

    let model = BertModel::load(vb, &config).expect("BertModel failed.");

    let mut npz = NpzReader::new(File::open("static/pkg/constants.npz")?)?;
    let mask : Array1<bool> = npz.by_name("MASK")?;
    let r_reg_fus : Array2<f32> = npz.by_name("R_REG_FUS")?;
    let l_reg_fus : Array2<f32> = npz.by_name("L_REG_FUS")?;

    Ok((model, tokenizer.unwrap(), mask, l_reg_fus, r_reg_fus, title_embeddings))
}

pub fn text_query(
    query: &str,
    model: &BertModel,
    tokenizer: &mut Tokenizer,
    title_embeddings: &Tensor,
    mask: &Array1<bool>,
    l_reg_fus: &Array2<f32>,
    r_reg_fus: &Array2<f32>
) -> Result<Vec<f32>, Box<dyn std::error::Error>>{

    // Compute embedding for the query
    let sentences = [
       query
    ];

    let tokens = tokenizer
        .encode_batch(sentences.to_vec(), true)
        .map_err(E::msg).unwrap();

    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    } else {
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }

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

    let query_embedding = embeddings.get(0).expect("Failed to unpack embeddings");
    let n_titles = title_embeddings.dim(0).expect("Failed to get number of tiles");

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

    // Projections
    //   unpack aligner
    let aligner_data = read("static/pkg_encoder/assets/aligner.safetensors")?;
    let aligner_tensors = load_buffer(&aligner_data, &device)?;
    let aligner_w0 = aligner_tensors.get("align_w0").ok_or("align_w0 tensor not found")?;
    let aligner_b0 = aligner_tensors.get("align_b0").ok_or("align_b0 tensor not found")?;
    let aligner_w1 = aligner_tensors.get("align_w1").ok_or("align_w1 tensor not found")?;
    let aligner_b1 = aligner_tensors.get("align_b1").ok_or("align_b1 tensor not found")?;
    // .to_dtype(DType::F32)

    //   unpack decoder
    let decoder_data = read("static/pkg_encoder/assets/decoder.safetensors")?;
    let decoder_tensors = load_buffer(&decoder_data, &device)?;
    let decoder_w0 = decoder_tensors.get("decoder_w0").ok_or("decoder_w0 tensor not found")?;
    let decoder_b0 = decoder_tensors.get("decoder_b0").ok_or("decoder_b0 tensor not found")?;
    let decoder_w1 = decoder_tensors.get("decoder_w1").ok_or("decoder_w1 tensor not found")?;
    let decoder_b1 = decoder_tensors.get("decoder_b1").ok_or("decoder_b1 tensor not found")?;
    let decoder_w2 = decoder_tensors.get("decoder_w2").ok_or("decoder_w2 tensor not found")?;
    let decoder_b2 = decoder_tensors.get("decoder_b2").ok_or("decoder_b2 tensor not found")?;

    // //  weight titles based on similarity to query
    // let top_k : usize = 10;
    // let mut top_inds : Vec<i64> = vec![0; top_k];
    // let mut title_weights : Vec<f32> = vec![0.0; top_k];
    // let mut w_sum : f32 = 0.0;

    // for i in 0..top_k{
    //     title_weights[i] = similarities[i].0;
    //     top_inds[i] = similarities[i].1 as i64;
    //     w_sum = w_sum + title_weights[i];
    // }

    // for i in 0..top_k{
    //     title_weights[i] = title_weights[i] / w_sum;
    // }

    // let indices_tensor = Tensor::from_vec(top_inds.clone(), (top_inds.len(),), &Device::Cpu)
    //     .expect("from_vec error.");

    // let title_embeddings_topk = title_embeddings
    //     .index_select(&indices_tensor, 0 as usize)
    //     .expect("Index error.");

    // let tw =  Tensor::from_vec(title_weights, (1, top_k), &device)
    //     .expect("From vec error.");

    //  let tw = title_embeddings_topk;

    // let title_vec = tw.broadcast_matmul(
    //     &title_embeddings_topk
    // ).expect("Shape mismatch.");

    // let l2_norm_sq = title_vec
    //     .sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();

    // let query_embedding_reshaped = query_embedding.unsqueeze(1).unwrap();
    // let mut _proj = title_vec.matmul(&query_embedding_reshaped).unwrap().squeeze(1).unwrap();

    // let _title_vec = title_vec.get(0).unwrap();
    // let constant = _proj.get(0).unwrap().to_scalar::<f32>().unwrap() / l2_norm_sq;

    // let constant_vec = vec![constant; _title_vec.shape().dims()[0]];
    // let constant_tensor = Tensor::new(constant_vec, &device).unwrap();
    // let proj = (&_title_vec * &constant_tensor).unwrap().unsqueeze(0).unwrap();

    //   push latent vector through aligner network
    // let x0 = (
    //     proj.broadcast_matmul(&aligner_w0.t().unwrap()) + aligner_b0.unsqueeze(0).unwrap()
    // ).unwrap().relu().unwrap();
    let x0 = (
        query_embedding.broadcast_matmul(&aligner_w0.t().unwrap()) + aligner_b0.unsqueeze(0).unwrap()
    ).unwrap().relu().unwrap();

    let neuro_aligned = (
        x0.broadcast_matmul(&aligner_w1.t().unwrap()) + aligner_b1.unsqueeze(0).unwrap()
    ).unwrap();

    //  push through decoder network
    let x0 = (
        neuro_aligned.matmul(&decoder_w0.t().unwrap()).unwrap()
        + decoder_b0.unsqueeze(0).unwrap()
    ).unwrap().relu().unwrap();

    let x1 = (
        x0.matmul(&decoder_w1.t().unwrap()).unwrap()
        + decoder_b1.unsqueeze(0).unwrap()
    ).unwrap().relu().unwrap();

    let x2 = (
        x1.matmul(&decoder_w2.t().unwrap()).unwrap()
        + decoder_b2.unsqueeze(0).unwrap()
    ).unwrap();

    let neuro_pred = sigmoid(&x2).unwrap().reshape((28542,)).unwrap();

    // Transform to surface
    //   read from global variables
    let mut img3d: Array3<f32> = Array3::zeros((46, 55, 46));
    let mut ii : usize = 0;
    let mut jj : usize = 0;
    for i in 0..46{
        for j in 0..55{
            for k in 0..46{
                if mask[jj]{
                    img3d[[i, j, k]] = neuro_pred.get(ii).unwrap().to_scalar::<f32>().unwrap();
                    ii = ii + 1;
                }
                jj = jj+ 1;
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

    //  Concat
    let mut out : Vec<f32> = vec![0.0; 327884]; // 327684
    out[..327684].copy_from_slice(&lsurface_vec);
    for i in 327684..327784{
        out[i] = similarities[i-327684].0;
    }
    for i in 327784..327884{
        out[i] = similarities[i-327784].1 as f32;
    }
    Ok(out)
}
