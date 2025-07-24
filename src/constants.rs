// Load constant vectors and matrices
use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use std::error::Error;
use std::io::{Read, Seek};

#[derive(Clone)]
pub struct Constants{
    pub mask: Array1<bool>,
    pub r_reg_fus: Array2<f32>,
    pub l_reg_fus: Array2<f32>,
}

impl Default for Constants {
    fn default() -> Self {
        Constants {
            mask: Array1::from_elem(0, false),
            r_reg_fus: Array2::zeros((0, 0)),
            l_reg_fus: Array2::zeros((0, 0)),
        }
    }
}

impl Constants {
    pub fn new<R: Read + Seek>(mut npz: NpzReader<R>) -> Result<Self, Box<dyn Error>> {

        let mask : Array1<bool> = npz.by_name("MASK")?;
        let r_reg_fus : Array2<f32> = npz.by_name("R_REG_FUS")?;
        let l_reg_fus : Array2<f32> = npz.by_name("L_REG_FUS")?;

        Ok(Constants {
            mask: mask,
            r_reg_fus: r_reg_fus,
            l_reg_fus: l_reg_fus,
        })
    }
}