extern crate opencv;
extern crate ndarray;
extern crate ndarray_linalg as linalg;
use opencv::core::{Mat, Rect, Scalar, Size, BORDER_CONSTANT};
use opencv::imgproc::{warp_affine, resize, INTER_LINEAR};
use opencv::prelude::{MatTraitConst};
use opencv::calib3d::{estimate_affine_partial_2d, LMEDS};
use anyhow::{Error, Result};
use ndarray::{Array1, Array2};
use crate::pipeline::utils::utils::array2_to_mat;

#[derive(Debug, Clone)]
pub(crate) struct FaceAlignment {
    image_size: (i32, i32),
    standard_landmarks: Array2<f32>,
}

impl FaceAlignment {
    pub fn new(image_size: (i32, i32), standard_landmarks: Array2<f32>) -> Self {
        FaceAlignment {
            image_size,
            standard_landmarks,
        }
    }

    pub fn call(&self, img: &Mat, bbox: Option<Array1<f32>>, landmarks: Option<Array2<f32>>) -> Result<Mat, Error> {

        let img_shape = match img.size() {
            Ok(img_shape) => img_shape,
            Err(e) => return Err(Error::from(e))
        };

        let standard_landmarks_mat = match array2_to_mat(&self.standard_landmarks) {
            Ok(standard_landmarks_mat) => {standard_landmarks_mat}
            Err(e) => return Err(Error::from(e)),
        };
        let mut landmarks_mat = Mat::default();
        if let Some(_landmarks) = landmarks {
            landmarks_mat = match array2_to_mat(&_landmarks) {
                Ok(landmarks_mat) => {landmarks_mat}
                Err(e) => return Err(Error::from(e))
            };
        }

        let mut inliers = Mat::default();

        let transformation_matrix = match estimate_affine_partial_2d(
            &landmarks_mat,
            &standard_landmarks_mat,
            &mut inliers,
            LMEDS,
            3.0,
            2000,
            0.99,
            10,
        ) {
            Ok(transformation_matrix) => {transformation_matrix},
            Err(e) => return Err(Error::from(e)),
        };

        drop(landmarks_mat);
        drop(standard_landmarks_mat);
        drop(inliers);

        if transformation_matrix.empty() {
            let mut det: Array1<f32> = Array1::zeros(4);
            if bbox.is_none() {
                det = Array1::zeros(4);
                det[0] = img_shape.width.to_owned() as f32 * 0.0625;
                det[1] = img_shape.height as f32 * 0.0625;
                det[2] = img_shape.width as f32 - det[0];
                det[3] = img_shape.height as f32 - det[1];
            } else {
                det = bbox.unwrap();
            }

            let margin: f32 = 44.0;
            let mut bb: Array1<f32> = Array1::zeros(4);
            bb[0] = f32::max(det[0] - margin / 2.0, 0.0);
            bb[1] = f32::max(det[1] - margin / 2.0, 0.0);
            bb[2] = f32::max(det[2] + margin / 2.0, img_shape.width as f32);
            bb[3] = f32::max(det[1] + margin / 2.0, img_shape.height as f32);

            let x0 = bb[0] as i32;
            let y0 = bb[1] as i32;
            let x1 = bb[2] as i32;
            let y1 = bb[3] as i32;
            let width = x1 - x0;
            let height = y1 - y0;

            let rect = Rect::new(x0, y0, width, height);
            let mut cropped_img = img.clone();
            let roi = match Mat::roi_mut(&mut cropped_img, rect) {
                Ok(roi) => {roi}
                Err(e) => return Err(Error::from(e)),
            };

            let mut resized_image = Mat::default();
            match resize(
                &roi,
                &mut resized_image,
                Size::new(self.image_size.0, self.image_size.1),
                0.0,
                0.0,
                INTER_LINEAR,
            ){
                Ok(_) => {
                    drop(cropped_img);
                    Ok(resized_image)
                },
                Err(e) => {
                    drop(cropped_img);
                    return Err(Error::from(e))
                },
            }
        } else {
            let mut aligned_image = Mat::default();
            match warp_affine(
                &img,
                &mut aligned_image,
                &transformation_matrix,
                Size::new(self.image_size.0, self.image_size.1),
                INTER_LINEAR,
                BORDER_CONSTANT,
                Scalar::default())
            {
                Ok(_) => {
                    drop(transformation_matrix);
                    Ok(aligned_image)

                },
                Err(e) => {
                    drop(transformation_matrix);
                    return Err(Error::from(e))
                },
            }
        }
    }
}


#[cfg(test)]
mod tests {

    #[tokio::test]
    async fn test_face_alignment() {
    }
}