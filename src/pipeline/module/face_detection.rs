use opencv::core::{self, Mat, MatTraitConst, Scalar, Size};
use opencv::imgproc::{INTER_LINEAR, resize};
use std::collections::HashMap;
use std::ops::{MulAssign};
use crate::pipeline::processing::generate_anchors::{AnchorConfig, Config, generate_anchors_fpn2};
use crate::pipeline::triton_client::client::TritonInferenceClient;
use anyhow::{Error, Result};
use ndarray::{Array, Array2, Array3, Array4, ArrayBase, Axis, concatenate, Dim, IntoDimension, Ix, Ix2, Ix3, OwnedRepr, s};
use opencv::imgcodecs::imwrite;
use crate::pipeline::processing::bbox_transform::clip_boxes;
use crate::pipeline::processing::nms::nms;
use crate::pipeline::rcnn::anchors::anchors;
use crate::pipeline::triton_client::client::triton::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use crate::pipeline::triton_client::client::triton::{InferParameter, InferTensorContents, ModelConfigRequest, ModelConfigResponse, ModelInferRequest};
use crate::pipeline::triton_client::client::triton::infer_parameter::ParameterChoice;
use crate::pipeline::utils::utils::{argsort_descending, reorder_2d, reorder_3d, u8_to_f32_vec, vstack_2d, vstack_3d};

#[derive(Debug, Clone)]
pub struct RetinaFaceDetection {
    triton_infer_client: TritonInferenceClient,
    triton_model_config: ModelConfigResponse,
    model_name: String,
    max_batch_size: i32,
    image_size: (i32, i32),
    use_landmarks: bool,
    confidence_threshold: f32,
    iou_threshold: f32,
    fpn_keys: Vec<String>,
    _feat_stride_fpn: Vec<i32>,
    anchor_cfg: HashMap<String, AnchorConfig>,
    _anchors_fpn: HashMap<String, Array2<f32>>,
    _num_anchors: HashMap<String, usize>,
    pixel_means: Vec<f32>,
    pixel_stds: Vec<f32>,
    pixel_scale: f32,
    bbox_stds: Vec<f32>,
    landmark_std: f32,
}

impl RetinaFaceDetection {
    pub async fn new(
        triton_infer_client: TritonInferenceClient,
        triton_model_config: ModelConfigResponse,
        model_name: String,
        image_size: (i32, i32),
        max_batch_size: i32,
        confidence_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Self, Error> {

        let mut fpn_keys = vec![];
        let _feat_stride_fpn = vec![32, 16, 8];
        let _ratio = vec![1.0];

        let mut anchor_cfg = HashMap::new();
        anchor_cfg.insert(
            "32".to_string(), AnchorConfig {
                scales: vec![32.0, 16.0],
                base_size: 16,
                ratios: _ratio.clone(),
                allowed_border: 9999
            }
        );
        anchor_cfg.insert(
            "16".to_string(),
            AnchorConfig {
                base_size: 16,
                ratios: _ratio.clone(),
                scales: vec![8.0, 4.0],
                allowed_border: 9999,
            },
        );
        anchor_cfg.insert(
            "8".to_string(), AnchorConfig {
                scales: vec![2.0, 1.0],
                base_size: 16,
                ratios: _ratio.clone(),
                allowed_border: 9999
            }
        );
        let config = Config {
            rpn_anchor_cfg: anchor_cfg.clone(),
        };

        for s in &_feat_stride_fpn {
            fpn_keys.push(format!("stride{}", s));
        }

        let dense_anchor = false;
        let use_landmarks= true;
        let bbox_stds= vec![1.0, 1.0, 1.0, 1.0];
        let landmark_std = 1.0;

        let _anchors_fpn = fpn_keys
            .iter()
            .zip(generate_anchors_fpn2(dense_anchor, Some(&config)))
            .map(|(k, v)| (k.clone(), v))
            .collect::<HashMap<_, _>>();

        let _num_anchors = _anchors_fpn
            .iter()
            .map(|(k, v)| (k.clone(), v.clone().shape()[0]))
            .collect::<HashMap<_, _>>();

        let pixel_means = vec![0.0, 0.0, 0.0];
        let pixel_stds = vec![1.0, 1.0, 1.0];
        let pixel_scale = 1.0;

        Ok(RetinaFaceDetection {
            triton_infer_client,
            triton_model_config,
            model_name,
            image_size,
            use_landmarks,
            confidence_threshold,
            iou_threshold,
            fpn_keys,
            _feat_stride_fpn,
            anchor_cfg,
            _anchors_fpn,
            _num_anchors,
            pixel_means,
            pixel_stds,
            pixel_scale,
            bbox_stds,
            landmark_std,
            max_batch_size,
        })
    }

    fn _preprocess(&self, img: Mat) -> Result<(Mat, f32), Error> {

        let img_shape = match img.size() {
            Ok(img_shape) => img_shape,
            Err(e) => return Err(Error::from(e))
        };

        let im_ratio = img_shape.height as f32 / img_shape.width as f32;
        let model_ratio = self.image_size.1 as f32 / self.image_size.0 as f32;

        let (new_width, new_height) = if im_ratio > model_ratio {
            let new_height = self.image_size.1;
            let new_width = (new_height as f32 / im_ratio) as i32;
            (new_width, new_height)
        } else {
            let new_width = self.image_size.0;
            let new_height = (new_width as f32 * im_ratio) as i32;
            (new_width, new_height)
        };

        let det_scale = new_height as f32 / img_shape.height as f32;
        let mut resized_img = Mat::default();

        match resize(&img, &mut resized_img, Size::new(new_width, new_height), 0.0, 0.0, INTER_LINEAR) {
            Ok(_) => {},
            Err(e) => return Err(Error::from(e))
        }


        let mut det_img = match Mat::new_rows_cols_with_default(self.image_size.1, self.image_size.0, core::CV_8UC3, Scalar::all(0.0)) {
            Ok(det_img) => det_img,
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        let  mut roi = match Mat::roi_mut(&mut det_img, core::Rect::new(0, 0, new_width, new_height)) {
            Ok(roi) => roi,
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        match  resized_img.copy_to(&mut roi) {
            Ok(_) => {},
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        drop(resized_img);
        drop(img);

        Ok((det_img, det_scale))
    }


    async fn _forward(&self, img: Mat) -> Result<(Array2<f32>, Option<Array3<f32>>), Error> {

        let im_info = match img.size() {
            Ok(im_info) => {im_info}
            Err(e) => return Err(Error::from(e))
        };
        let rows = im_info.height;
        let cols = im_info.width;

        let mut im_tensor = Array4::<f32>::zeros((1, 3, rows as usize, cols as usize));

        // Convert the image to float and normalize it
        for i in 0..3 {
            for y in 0..rows {
                for x in 0..cols {
                    let pixel_value = img.at_2d::<core::Vec3b>(y, x).unwrap()[2 - i];
                    im_tensor[[0, i, y as usize, x as usize]] = (pixel_value as f32 / self.pixel_scale - self.pixel_means[2 - i]) / self.pixel_stds[2 - i];
                }
            }
        }

        let vec = im_tensor.into_raw_vec();

        let model_cfg = match &self.triton_model_config.config {
            None => {
                return Err(Error::msg("face_detection - face detection model config is empty"))
            }
            Some(model_cfg) => {model_cfg}
        };

        let mut cfg_outputs = Vec::<InferRequestedOutputTensor>::new();

        for out_cfg in model_cfg.output.iter() {
            cfg_outputs.push(InferRequestedOutputTensor {
                name: out_cfg.name.to_string(),
                parameters: Default::default(),
            })
        }


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
                contents: Some(InferTensorContents {
                    bool_contents: vec![],
                    int_contents: vec![],
                    int64_contents: vec![],
                    uint_contents: vec![],
                    uint64_contents: vec![],
                    fp32_contents: vec,
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

        let mut net_out: Vec<Array4<f32>> = vec![Array4::zeros([1,1,1,1]); cfg_outputs.len()];

        for (idx, output) in &mut model_out.outputs.iter_mut().enumerate() {
            let dims = &output.shape;
            let dimensions: [usize; 4] = [
                dims[0] as usize,
                dims[1] as usize,
                dims[2] as usize,
                dims[3] as usize,
            ];
            let u8_array: &[u8] = &model_out.raw_output_contents[idx];
            let f_array = u8_to_f32_vec(u8_array);

            let array4_f32: Array4<f32> = match Array4::from_shape_vec(dimensions.into_dimension(), f_array) {
                Ok(array4_f32) => {array4_f32}
                Err(e) => {
                    return Err(Error::from(e))
                }
            };
            let result_index = match cfg_outputs.iter().position(|r| *r.name == output.name) {
                None => {
                    return Err(Error::msg("face_detection - no matched model index"))
                }
                Some(result_index) => {result_index}
            };
            net_out[result_index] =  array4_f32;
        }

        drop(model_out);

        let mut proposals_list: Vec<Array2<f32>> = Vec::new();
        let mut scores_list: Vec<Array<f32, Dim<[Ix; 2]>>> = Vec::new();
        let mut landmarks_list: Vec<Array3<f32>> = Vec::new();
        let mut sym_idx = 0;

        for s in &self._feat_stride_fpn {
            let stride = *s as usize;
            let scores = net_out[sym_idx].to_owned();
            let sliced_scores = &scores.slice(s![.., self._num_anchors[&format!("stride{}", stride)].., .., ..]).to_owned();
            let bbox_deltas = net_out[sym_idx + 1].to_owned();
            let height = bbox_deltas.shape()[2];
            let width = bbox_deltas.shape()[3];


            let a = self._num_anchors[&format!("stride{}", stride)];
            let k = height * width;
            let anchors_fpn = &self._anchors_fpn[&format!("stride{}", stride)];
            let anchor_plane = anchors(height, width, stride, anchors_fpn);
            let anchors_reshape = match anchor_plane.into_shape((k * &a, 4)) {
                Ok(anchors_reshape) => {anchors_reshape}
                Err(e) => {
                    return Err(Error::from(e))
                }
            };

            let transposed_scores = sliced_scores.clone().permuted_axes([0, 2, 3, 1]);
            let scores_shape = transposed_scores.shape();
            let mut scores_dim: usize = 1;
            for dim in scores_shape {
                scores_dim *= dim;
            }
            let flattened_scores: Vec<f32> = transposed_scores.iter().cloned().collect();
            drop(transposed_scores);


            let arr_scores = Array::from(flattened_scores);
            let reshaped_scores = match arr_scores.into_shape((scores_dim, 1)) {
                Ok(reshaped_scores) => {reshaped_scores}
                Err(e) => {
                    return Err(Error::from(e))
                }
            };
            let transposed_bbox_deltas = bbox_deltas.permuted_axes([0, 2, 3, 1]);
            let bbox_pred_len = transposed_bbox_deltas.dim().3 / &a;
            let bbox_deltas_shape = transposed_bbox_deltas.shape();
            let mut bbox_deltas_dim: usize = 1;
            for dim in bbox_deltas_shape {
                bbox_deltas_dim *= dim;
            }
            let flattened_bbox_deltas: Vec<f32> = transposed_bbox_deltas.iter().cloned().collect();
            let arr_bbox_deltas = Array::from(flattened_bbox_deltas);
            let mut bbox_deltas_reshaped = match arr_bbox_deltas.into_shape(((bbox_deltas_dim) / &bbox_pred_len, bbox_pred_len)) {
                Ok(bbox_deltas_reshaped) => {bbox_deltas_reshaped}
                Err(e) => {
                    return Err(Error::from(e))
                }
            };
            drop(transposed_bbox_deltas);

            for i in (0..4).step_by(4) {
                bbox_deltas_reshaped.slice_mut(s![.., i]).mul_assign(self.bbox_stds[i]);
                bbox_deltas_reshaped.slice_mut(s![.., i + 1]).mul_assign(self.bbox_stds[i + 1]);
                bbox_deltas_reshaped.slice_mut(s![.., i + 2]).mul_assign(self.bbox_stds[i + 2]);
                bbox_deltas_reshaped.slice_mut(s![.., i + 3]).mul_assign(self.bbox_stds[i + 3]);
            }
            let mut proposals = self.bbox_pred(anchors_reshape.clone(), bbox_deltas_reshaped.to_owned());
            drop(bbox_deltas_reshaped);

            clip_boxes(&mut proposals, (im_info.height as usize, im_info.width as usize));
            let scores_flatten = reshaped_scores.view().iter().copied().collect::<Vec<_>>();
            let order: Vec<usize> = scores_flatten.iter().enumerate().filter(|(_, &s)| s >= self.confidence_threshold).map(|(i, _)| i).collect();
            let selected_proposals = proposals.select(Axis(0), &order);
            let selected_scores = reshaped_scores.select(Axis(0), &order);
            proposals_list.push(selected_proposals);
            scores_list.push(selected_scores);

            if self.use_landmarks {
                let landmark_deltas = net_out[sym_idx + 2].to_owned();
                let landmark_pred_len = landmark_deltas.dim().1 / a;
                let transposed_landmark_deltas = &landmark_deltas.permuted_axes([0, 2, 3, 1]);
                let landmark_deltas_shape = transposed_landmark_deltas.shape();
                let mut landmark_deltas_dim: usize = 1;
                for dim in landmark_deltas_shape {
                    landmark_deltas_dim *= dim;
                }
                let flattened_landmark_deltas: Vec<f32> = transposed_landmark_deltas.iter().cloned().collect();
                let arr_landmark_deltas = Array::from(flattened_landmark_deltas);
                let mut reshaped_landmark_deltas = match arr_landmark_deltas.into_shape((landmark_deltas_dim / landmark_pred_len, 5, landmark_pred_len / 5)) {
                    Ok(reshaped_landmark_deltas) => {reshaped_landmark_deltas}
                    Err(e) => {
                        return Err(Error::from(e))
                    }
                };
                reshaped_landmark_deltas *= self.landmark_std;
                let landmarks = self.landmark_pred(anchors_reshape, reshaped_landmark_deltas);
                let selected_landmarks = landmarks.select(Axis(0), &order);
                landmarks_list.push(selected_landmarks);
                drop(landmarks);
            }
            if self.use_landmarks {
                sym_idx += 3;
            } else {
                sym_idx += 2;
            }
        }

        drop(net_out);

        let proposals = vstack_2d(proposals_list);
        let mut landmarks: Option<Array3<f32>> = None;

        if proposals.dim().0 == 0 {
            if self.use_landmarks {
                let det: Array2<f32> = Array2::zeros((0, 5));
                landmarks = Some(Array3::zeros((0, 5, 2)));
                return Ok((det, landmarks));
            }
        }

        let score_stack = vstack_2d(scores_list);
        let score_flatten = score_stack.view().iter().copied().collect::<Vec<_>>();


        let order = argsort_descending(&score_flatten);
        let selected_proposals = reorder_2d(proposals, &order);
        let selected_score = reorder_2d(score_stack, &order);
        if self.use_landmarks {
            let landmarks_stack = vstack_3d(landmarks_list);
            landmarks = Some(reorder_3d(landmarks_stack, &order));
        }
        drop(order);

        let pre_det = concatenate![Axis(1), selected_proposals.slice(s![.., 0..4]).to_owned(), selected_score];
        let keep = nms(&pre_det, self.iou_threshold);
        let det = concatenate![Axis(1), pre_det, selected_proposals.slice(s![.., 4..]).to_owned()];
        let selected_rows = &keep.iter()
            .map(|&i| det.slice(s![i, ..]).to_owned())
            .collect::<Vec<_>>();
        let new_det_shape = (selected_rows.len(), det.shape()[1]);
        let det = match Array2::from_shape_vec(
            new_det_shape,
            selected_rows.into_iter().flat_map(| row| row.iter().cloned()).collect()
        ) {
            Ok(det) => {det}
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        if self.use_landmarks {
            let selected_landmarks = &keep.iter()
                .map(|&i| landmarks.as_mut().unwrap().slice(s![i, .., ..]).to_owned())
                .collect::<Vec<_>>();

            let new_landmarks_shape = (selected_landmarks.len(), landmarks.as_mut().unwrap().shape()[1], landmarks.as_mut().unwrap().shape()[2]);

            landmarks = match Array3::from_shape_vec(
                new_landmarks_shape,
                selected_landmarks.into_iter().flat_map(|array| array.iter().cloned()).collect()
            ) {
                Ok(landmarks) => {
                    Some(landmarks)
                }
                Err(e) => {
                    return Err(Error::from(e))
                }
            };
        }
        Ok((det, landmarks))
    }

    async fn _postprocess(&self, predicted_output: (Array2<f32>, Option<Array3<f32>>), preprocess_param: f32) -> (Array2<f32>, Option<Array3<f32>>)  {
        let (mut det, mut kpss) = predicted_output;
        for mut row in det.axis_iter_mut(Axis(0)) {
            for elem in row.iter_mut().take(4) {
                *elem /= preprocess_param;
            }
        }
        if !kpss.is_none() {
            kpss.as_mut().unwrap().map_inplace(|x| *x /= preprocess_param);
        }
        (det, kpss)
    }


    pub async fn call(&self, image: Mat) -> Result<(Array2<f32>, Option<Array3<f32>>), Error> {

        let (preprocessed_img, scale) = match self._preprocess(image) {
            Ok((preprocessed_img, scale)) => {(preprocessed_img, scale)}
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        let predicted_output = match self._forward(preprocessed_img).await {
            Ok(predicted_output) => {predicted_output}
            Err(e) => {
                return Err(Error::from(e))
            }
        };
        let (det, landmarks) = self._postprocess(predicted_output, scale).await;

        Ok((det, landmarks))
    }


    fn bbox_pred(&self, boxes: ArrayBase<OwnedRepr<f32>, Ix2>, box_deltas: ArrayBase<OwnedRepr<f32>, Ix2>) -> Array2<f32> {
        if boxes.shape()[0] == 0 {
            return Array2::zeros((0, box_deltas.shape()[1]));
        }

        let boxes = boxes.mapv(|x| x as f32);
        let widths = &boxes.slice(s![.., 2]) - &boxes.slice(s![.., 0]) + 1.0;
        let heights = &boxes.slice(s![.., 3]) - &boxes.slice(s![.., 1]) + 1.0;
        let ctr_x = &boxes.slice(s![.., 0]) + 0.5 * (&widths - 1.0);
        let ctr_y = &boxes.slice(s![.., 1]) + 0.5 * (&heights - 1.0);

        let dx = box_deltas.slice(s![.., 0..1]);
        let dy = box_deltas.slice(s![.., 1..2]);
        let dw = box_deltas.slice(s![.., 2..3]);
        let dh = box_deltas.slice(s![.., 3..4]);

        let pred_ctr_x = &dx * &widths.clone().insert_axis(Axis(1)) + &ctr_x.insert_axis(Axis(1));
        let pred_ctr_y = &dy * &heights.clone().insert_axis(Axis(1)) + &ctr_y.insert_axis(Axis(1));
        let pred_w = dw.mapv(f32::exp) * &widths.insert_axis(Axis(1));
        let pred_h = dh.mapv(f32::exp) * &heights.insert_axis(Axis(1));

        let mut pred_boxes = Array2::<f32>::zeros(box_deltas.raw_dim());

        pred_boxes.slice_mut(s![.., 0..1]).assign(&(&pred_ctr_x - 0.5 * (&pred_w - 1.0)));
        pred_boxes.slice_mut(s![.., 1..2]).assign(&(&pred_ctr_y - 0.5 * (&pred_h - 1.0)));
        pred_boxes.slice_mut(s![.., 2..3]).assign(&(&pred_ctr_x + 0.5 * (&pred_w - 1.0)));
        pred_boxes.slice_mut(s![.., 3..4]).assign(&(&pred_ctr_y + 0.5 * (&pred_h - 1.0)));

        drop(pred_w);
        drop(pred_h);
        drop(pred_ctr_x);
        drop(pred_ctr_y);

        if box_deltas.shape()[1] > 4 {
            pred_boxes.slice_mut(s![.., 4..]).assign(&box_deltas.slice(s![.., 4..]));
        }

        pred_boxes
    }

    fn landmark_pred(&self, boxes: ArrayBase<OwnedRepr<f32>, Ix2>, landmark_deltas: ArrayBase<OwnedRepr<f32>, Ix3>) -> ArrayBase<OwnedRepr<f32>, Ix3> {
        if boxes.shape()[0] == 0 {
            return Array3::zeros((0, landmark_deltas.shape()[1], landmark_deltas.shape()[2]));
        }

        let boxes = boxes.mapv(|x| x);
        let widths = &boxes.slice(s![.., 2]) - &boxes.slice(s![.., 0]) + 1.0;
        let heights = &boxes.slice(s![.., 3]) - &boxes.slice(s![.., 1]) + 1.0;
        let ctr_x = &boxes.slice(s![.., 0]) + 0.5 * (&widths - 1.0);
        let ctr_y = &boxes.slice(s![.., 1]) + 0.5 * (&heights - 1.0);

        let mut pred = landmark_deltas.clone();

        for i in 0..5 {
            pred.slice_mut(s![.., i, 0]).assign(&(&landmark_deltas.slice(s![.., i, 0]) * &widths + &ctr_x));
            pred.slice_mut(s![.., i, 1]).assign(&(&landmark_deltas.slice(s![.., i, 1]) * &heights + &ctr_y));
        }

        pred
    }
}

#[cfg(test)]
mod tests {
    use crate::pipeline::module::face_detection::RetinaFaceDetection;
    use crate::pipeline::triton_client::client::triton::ModelConfigRequest;
    use crate::pipeline::triton_client::client::TritonInferenceClient;
    use crate::pipeline::utils::utils::byte_data_to_opencv;

    #[tokio::test]
    async fn test_retina_face_detection() {
        // let triton_host = "";
        // let triton_port = "";
        // let im_bytes: &[u8] = include_bytes!("");
        // let image = byte_data_to_opencv(im_bytes).unwrap();
        //
        // let triton_infer_client = match TritonInferenceClient::new(triton_host, triton_port).await {
        //     Ok(triton_infer_client) => triton_infer_client,
        //     Err(e) => {
        //         println!("{:?}", e);
        //         return
        //     }
        // };
        //
        // let model_name = "face_detection_retina".to_string();
        //
        // let face_detection_model_config = match triton_infer_client
        //     .model_config(ModelConfigRequest {
        //         name: model_name.to_owned(),
        //         version: "".to_string(),
        //     }).await {
        //     Ok(model_config_resp) => {model_config_resp}
        //     Err(e) => {
        //         println!("{:?}", e);
        //         return
        //     }
        // };
        //
        // let retina_face_detection = match RetinaFaceDetection::new(
        //     triton_infer_client,
        //     face_detection_model_config,
        //     model_name,
        //     (640, 640),
        //     1,
        //     0.7,
        //     0.45,
        // ).await
        // {
        //     Ok(retina_face_detection)  => retina_face_detection,
        //     Err(e) => {
        //         println!("{:?}", e);
        //         return
        //     }
        // };
        //
        // let (preprocessed_img, scale) = retina_face_detection._preprocess(&image, Some(true)).unwrap();
        // let predicted_output = retina_face_detection._forward(&preprocessed_img, None).await.unwrap();
        // retina_face_detection._postprocess(predicted_output, scale, Some(true)).await;
    }
}
