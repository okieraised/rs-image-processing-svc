use std::collections::HashMap;
use anyhow::Error;
use ndarray::{Array2};
use opencv::core::Mat;
use opencv::imgcodecs::{imdecode, IMREAD_COLOR, imwrite};
use opencv::imgproc::{COLOR_BGR2RGB, cvt_color};
use crate::pipeline::utils::coordinate::Coord2D;

pub fn convert_image_to_ndarray(im_bytes: &[u8]) -> anyhow::Result<Mat, Error> {

    // Convert bytes to Mat
    let img_as_mat = match Mat::from_slice(im_bytes) {
        Ok(img_as_mat) => img_as_mat,
        Err(e) => {
            return Err(Error::from(e))
        }
    };

    // Decode the image
    let img_as_arr_bgr = match imdecode(&img_as_mat, IMREAD_COLOR) {
        Ok(img_as_arr_bgr) => img_as_arr_bgr,
        Err(e) => {
            return Err(Error::from(e))
        }
    };

    let mut img_as_arr_rgb = Mat::default();
    let _ = match cvt_color(&img_as_arr_bgr, &mut img_as_arr_rgb, COLOR_BGR2RGB, 0) {
        Ok(_) => {}
        Err(e) => return Err(Error::from(e))
    };

    match imwrite("./img_as_arr_rgb.png", &img_as_arr_rgb, &opencv::core::Vector::default()) {
        Ok(_) => {}
        Err(e) => return Err(Error::from(e))
    };

    Ok(img_as_arr_rgb)
}

pub fn convert_metadata_to_ndarray(metadata: HashMap<&str, Coord2D>) -> Result<Option<Array2<f32>>, Error> {

    if metadata.is_empty() {
        return Ok(None)
    }

    let mut result = Vec::new();
    let mut nrows = 0;
    let ncols = 2;

    let ordered_meta_key = vec!["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"];

    for key in ordered_meta_key {
        let val = metadata.get(key);
        if let Some(_val) = val {
            result.extend_from_slice(&*vec![_val.x, _val.y]);
            nrows += 1;
        }
    }

    let arr = match Array2::from_shape_vec((nrows, ncols), result) {
        Ok(arr) => {arr}
        Err(e) => return Err(Error::from(e))
    };

    Ok(Some(arr))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use crate::pipeline::utils::coordinate::Coord2D;
    use crate::pipeline::utils::image::{convert_image_to_ndarray, convert_metadata_to_ndarray};

    #[test]
    fn test_convert_image_to_ndarray() {
        let im_bytes: &[u8] = include_bytes!("../../test_data/cccd_1.jpg");

        match convert_image_to_ndarray(im_bytes) {
            Ok(img) => println!("Image loaded and converted successfully {:?}", img),
            Err(e) => eprintln!("Failed to load and convert image: {}", e),
        }
    }

    #[test]
    fn test_convert_metadata_to_ndarray() {

        let mut metadata: HashMap<&str, Coord2D> = HashMap::from(
            [
                ("left_eye", Coord2D{x: 169.7128, y: 213.38426 }),
                ("right_eye", Coord2D{x: 455.29285, y: 223.66956 }),
                ("nose", Coord2D{x: 310.71146, y: 320.74503 }),
                ("left_mouth", Coord2D{x: 195.21452, y: 379.8982 }),
                ("right_mouth", Coord2D{x: 408.377, y: 384.25134}),
            ]
        );

        let result = match convert_metadata_to_ndarray(metadata) {
            Ok(result) => {result}
            Err(e) => {
                println!("{:?}", e);
                return
            }
        };
        if result.is_some() {
            println!("{:?}", result);
        }
    }
}