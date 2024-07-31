use opencv::core::{self, Mat, MatTrait, MatTraitConst, Scalar};
use opencv::imgcodecs::{imdecode, IMREAD_UNCHANGED};
use opencv::imgproc::{cvt_color, COLOR_RGBA2RGB, COLOR_GRAY2RGB};
use anyhow::{Error, Result};
use ndarray::{Array2, Array3, ArrayBase, Axis, concatenate, Ix2, Ix3, OwnedRepr, s, stack};
use ndarray_linalg::Norm;

pub fn byte_data_to_opencv(im_bytes: &[u8]) -> Result<Mat, Error> {
    // Convert bytes to Mat
    let img_as_mat = match Mat::from_slice(im_bytes) {
        Ok(img_as_mat) => img_as_mat,
        Err(e) => {
            return Err(Error::from(e))
        }
    };

    // Decode the image
    let opencv_img = match imdecode(&img_as_mat, IMREAD_UNCHANGED) {
        Ok(opencv_img) => opencv_img,
        Err(e) => {
            return Err(Error::from(e))
        }
    };

    // Check the number of channels and convert if necessary
    let opencv_img = match opencv_img.channels() {
        4 => {
            let mut rgb_img = Mat::default();
            match cvt_color(&opencv_img, &mut rgb_img, COLOR_RGBA2RGB, 0) {
                Ok(_) => {},
                Err(e) => {
                    return Err(Error::from(e))
                }
            };
            rgb_img
        }
        2 => {
            let mut rgb_img = Mat::default();
            match cvt_color(&opencv_img, &mut rgb_img, COLOR_GRAY2RGB, 0) {
                Ok(_) => {
                },
                Err(e) => {
                    return Err(Error::from(e))
                }
            };
            rgb_img
        }
        _ => opencv_img,
    };

    Ok(opencv_img)
}

pub fn vstack_2d(v: Vec<ArrayBase<OwnedRepr<f32>, Ix2>>) -> Array2<f32> {
    // Check if proposals_list is empty
    if v.is_empty() {
        return Array2::<f32>::zeros((0, 0));
    }

    // Initialize an array with the first element
    let mut stacked_array = v[0].clone();

    // Iterate over the rest of the proposals and concatenate them vertically
    for e in &v[1..] {
        stacked_array = concatenate![Axis(0), stacked_array, e.clone()];
    }

    stacked_array
}

pub fn vstack_3d(v: Vec<ArrayBase<OwnedRepr<f32>, Ix3>>) -> Array3<f32> {
    // Check if landmarks_list is empty
    if v.is_empty() {
        return Array3::zeros((0, 0, 0));
    }

    // Collect views of all arrays in landmarks_list
    let mut stacked_array = v[0].clone();

    for e in &v[1..] {
        stacked_array = concatenate![Axis(0), stacked_array, e.clone()];
    }

    stacked_array
}

pub fn argsort_descending(scores_ravel: &Vec<f32>) -> Vec<usize> {
    // Create a vector of indices
    let mut indices: Vec<usize> = (0..scores_ravel.len()).collect();

    // Sort indices based on the values in scores_ravel, in descending order
    indices.sort_by(|&i, &j| scores_ravel[j].partial_cmp(&scores_ravel[i]).unwrap());

    indices
}

pub fn reorder_2d(proposals: ArrayBase<OwnedRepr<f32>, Ix2>, order: &Vec<usize>) -> Array2<f32> {
    // Create an empty vector to hold the reordered proposals
    let mut stacked_v = Vec::with_capacity(order.len());

    // Reorder the rows of proposals based on the order vector
    for &index in order {
        stacked_v.push(proposals.row(index).to_owned());
    }

    // Convert the vector of rows back to an Array2
    let stacked_v = stack(Axis(0), &stacked_v.iter().map(|row| row.view()).collect::<Vec<_>>()).unwrap();

    stacked_v
}

pub fn reorder_3d(v: Array3<f32>, order: &Vec<usize>) -> Array3<f32> {
    let num_landmarks = v.shape()[0];
    let height = v.shape()[1];
    let width = v.shape()[2];

    let mut reordered_v = Array3::<f32>::zeros((num_landmarks, height, width));

    for (new_idx, &original_idx) in order.iter().enumerate() {
        reordered_v.slice_mut(s![new_idx, .., ..]).assign(&v.slice(s![original_idx, .., ..]));
    }

    reordered_v
}

pub fn u8_to_f32_vec(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(TryInto::try_into)
        .map(Result::unwrap)
        .map(f32::from_le_bytes)
        .collect()
}

pub fn array2_to_mat(arr: &Array2<f32>) -> opencv::Result<Mat> {
    let rows = arr.shape()[0] as i32;
    let cols = arr.shape()[1] as i32;
    let mut mat = Mat::new_rows_cols_with_default(rows, cols, core::CV_32F, Scalar::all(0.0))?;

    for (i, row) in arr.outer_iter().enumerate() {
        for (j, val) in row.iter().enumerate() {
            *mat.at_2d_mut::<f32>(i as i32, j as i32)? = *val;
        }
    }

    Ok(mat)
}

pub fn normalize_outputs(outputs: Vec<Vec<Array2<f32>>>) -> Vec<Array2<f32>> {
    outputs.iter().map(|outer| {
        let array = &outer[0];
        let norm = array.norm_l2();
        array / norm
    }).collect()
}



#[cfg(test)]
mod tests {
    use crate::utils::utils::byte_data_to_opencv;

    #[test]
    fn test_nms() {
        let im_bytes: &[u8] = include_bytes!("../../test_data/anderson.jpg");

        match byte_data_to_opencv(im_bytes) {
            Ok(img) => println!("Image loaded and converted successfully {:?}", img),
            Err(e) => eprintln!("Failed to load and convert image: {}", e),
        }
    }

}