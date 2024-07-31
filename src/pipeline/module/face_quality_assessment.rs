use anyhow::Error;
use ndarray::{Array2, Array4, IntoDimension, s};
use opencv::core::{Mat, MatTraitConst, Size};
use opencv::imgproc::{COLOR_BGR2RGB, cvt_color, INTER_LINEAR, resize};
use crate::pipeline::triton_client::client::triton::{InferTensorContents, ModelConfigResponse, ModelInferRequest};
use crate::pipeline::triton_client::client::triton::model_infer_request::InferInputTensor;
use crate::pipeline::triton_client::client::TritonInferenceClient;
use crate::pipeline::utils::utils::u8_to_f32_vec;

#[derive(Debug, Clone)]
pub(crate) struct FaceQualityAssessment {
    triton_infer_client: TritonInferenceClient,
    triton_model_config: ModelConfigResponse,
    model_name: String,
    image_size: (i32, i32),
    threshold: f32,
    batch_size: i32,
}


impl FaceQualityAssessment {
    pub async fn new(
        triton_infer_client: TritonInferenceClient,
        triton_model_config: ModelConfigResponse,
        model_name: String,
        image_size: (i32, i32),
        batch_size: i32,
        threshold: f32,
    ) -> Result<Self, Error> {
        Ok(FaceQualityAssessment {
            triton_infer_client,
            triton_model_config,
            model_name,
            image_size,
            threshold,
            batch_size,
        })
    }

    pub async fn call(&self, images: Mat) -> Result<(Vec<f32>, Vec<i32>), Error>{

        let img_arr: Vec<Mat> = vec![images];

        let batch_size = img_arr.len();
        let mut scores: Vec<f32> = vec![];
        let mut idxs: Vec<i32> = vec![];

        for i in 0..batch_size {
            let mut resized_image = Mat::default();
            match resize(&img_arr[i], &mut resized_image,Size::new(self.image_size.0, self.image_size.1), 0.0, 0.0, INTER_LINEAR){
                Ok(_) => {},
                Err(e) => return Err(Error::from(e)),
            }

            let mut rgb_img = Mat::default();

            match cvt_color(&resized_image, &mut rgb_img, COLOR_BGR2RGB, 0) {
                Ok(_) => {}
                Err(e) => return Err(Error::from(e)),
            };

            let im_info = match rgb_img.size() {
                Ok(im_info) => {im_info}
                Err(e) => return Err(Error::from(e))
            };
            let rows = im_info.height;
            let cols = im_info.width;

            let mut im_tensor = Array4::<f32>::zeros((1, rows as usize, cols as usize, 3));

            // Convert the image to float and normalize it
            for i in 0..3 {
                for y in 0..rows {
                    for x in 0..cols {
                        let pixel_value = rgb_img.at_2d::<opencv::core::Vec3b>(y, x).unwrap()[i];
                        im_tensor[[0, y as usize, x as usize, i]] = (pixel_value as f32 - 127.5) * 0.00784313725;
                    }
                }
            }
            let transposed_tensor = im_tensor.permuted_axes([0, 3, 1, 2]);
            let flattened_vec: Vec<f32> = transposed_tensor.iter().cloned().collect();

            let model_cfg = match &self.triton_model_config.config {
                None => {
                    return Err(Error::msg("face_quality_assessment - face quality assessment model config is empty"))
                }
                Some(model_cfg) => {model_cfg}
            };

            let model_request = ModelInferRequest{
                model_name: self.model_name.to_owned(),
                model_version: "".to_string(),
                id: "".to_string(),
                parameters: Default::default(),
                inputs: vec![InferInputTensor {
                    name: model_cfg.input[0].name.to_string(),
                    datatype: model_cfg.input[0].data_type().as_str_name()[5..].to_uppercase(),
                    shape: model_cfg.input[0].dims.to_owned(),
                    parameters: Default::default(),
                    contents: Option::from(InferTensorContents {
                        bool_contents: vec![],
                        int_contents: vec![],
                        int64_contents: vec![],
                        uint_contents: vec![],
                        uint64_contents: vec![],
                        fp32_contents: flattened_vec,
                        fp64_contents: vec![],
                        bytes_contents: vec![],
                    }),
                }],
                outputs: Default::default(),
                raw_input_contents: vec![],
            };

            let mut model_out = match self.triton_infer_client.model_infer(model_request).await {
                Ok(model_out) => model_out,
                Err(e) => {
                    return Err(Error::from(e))
                }
            };

            let mut net_out: Vec<Array2<f32>> = vec![];

            for (idx, output) in &mut model_out.outputs.iter_mut().enumerate() {
                let dims = &output.shape;
                let dimensions: [usize; 2] = [
                    dims[0] as usize,
                    dims[1] as usize,
                ];
                let u8_array: &[u8] = &model_out.raw_output_contents[idx];
                let f_array = u8_to_f32_vec(u8_array);

                let array2_f32: Array2<f32> = match Array2::from_shape_vec(dimensions.into_dimension(), f_array) {
                    Ok(array2_f32) => {array2_f32}
                    Err(e) => {
                        return Err(Error::from(e))
                    }
                };
                net_out.push(array2_f32);
            }
            drop(model_out);

            let score = net_out[0].slice(s![0, 0]).into_scalar().to_owned();
            let predict = if score > self.threshold {
                1
            } else {
                0
            };

            idxs.push(predict);
            scores.push(score);
        }

        Ok((scores, idxs))
    }
}

#[cfg(test)]
mod tests {
}
