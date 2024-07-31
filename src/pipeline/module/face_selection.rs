use anyhow::Error;
use ndarray::{Array1, Array2, Array3};
use opencv::core::{Mat, MatTraitConst};

#[derive(Debug, Clone)]
pub struct FaceSelection {
    margin_center_left_ratio: f32,
    margin_center_right_ratio: f32,
    margin_edge_ratio: f32,
    minimum_face_ratio: f32,
}

impl FaceSelection {
    pub async fn new(
        margin_center_left_ratio: f32,
        margin_center_right_ratio: f32,
        margin_edge_ratio: f32,
        minimum_face_ratio: f32,
    ) -> Self {
        FaceSelection {
            margin_center_left_ratio,
            margin_center_right_ratio,
            margin_edge_ratio,
            minimum_face_ratio,
        }
    }

    fn get_biggest_area_face(&self, face_boxes: &Array2<f32>, key_points: &Option<Array3<f32>>) -> (Option<Array1<f32>>, Option<Array2<f32>>) {
        let mut biggest_area: f32 = 0.0;
        let mut biggest_bbox: Option<Array1<f32>> = None;
        let mut biggest_key_point: Option<Array2<f32>> = None;

        if let Some(kps) = key_points {
            for (bbox, key_point) in face_boxes.outer_iter().zip(kps.outer_iter()) {

                let (xmin, ymin, xmax, ymax, score) = (
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                    bbox[4],
                );

                if (xmax - xmin) * (ymax - ymin) > biggest_area {
                    biggest_area = (xmax - xmin) * (ymax - ymin);
                    biggest_bbox = Some(bbox.clone().to_owned());
                    biggest_key_point = Some(key_point.clone().to_owned());
                }
            }
        }
        (biggest_bbox, biggest_key_point)
    }

    fn is_face_area_big_enough(&self, img: &Mat, face_box: &Array1<f32>) -> Result<bool, Error> {
        let (xmin, ymin, xmax, ymax, _) = (
            face_box[0],
            face_box[1],
            face_box[2],
            face_box[3],
            face_box[4],
        );
        let img_shape = match img.size() {
            Ok(img_shape) => img_shape,
            Err(e) => return Err(Error::from(e))
        };
        let (image_height, image_width) = (img_shape.height as f32, img_shape.width as f32);
        let face_width = xmax - xmin;
        let face_height = ymax - ymin;
        Ok(face_width / image_width > 0.25)
    }

    pub fn call(&self, img: &Mat, face_boxes: Array2<f32>, key_points: Option<Array3<f32>>, is_enroll: Option<bool>) -> Result<(Option<Array1<f32>>, Option<Array2<f32>>), Error> {

        let enroll = is_enroll.unwrap_or(false);

        let img_shape = match img.size() {
            Ok(img_shape) => img_shape,
            Err(e) => return Err(Error::from(e))
        };

        if enroll {
            let (biggest_bbox, biggest_key_point) = self.get_biggest_area_face(&face_boxes, &key_points);

            if let Some(_biggest_bbox) = &biggest_bbox {
                let is_big_enough = match self.is_face_area_big_enough(img, &_biggest_bbox) {
                    Ok(is_big_enough) => {is_big_enough}
                    Err(e) => {
                        return Err(Error::from(e))
                    }
                };

                if is_big_enough {
                    return Ok((biggest_bbox, biggest_key_point))
                }
            }
            return Ok((biggest_bbox, biggest_key_point));
        }

        let margin_center_left = self.margin_center_left_ratio * img_shape.width as f32;
        let margin_center_right = self.margin_center_right_ratio * img_shape.width as f32;
        let mut margin_edge = self.margin_edge_ratio * img_shape.width as f32;
        margin_edge = f32::min(50.0, margin_edge);
        let y_cen = img_shape.height as f32 / 2.0;
        let x_cen = img_shape.width as f32 / 2.0;
        let mut valid_boxes: Vec<Vec<f32>> = Vec::with_capacity(1); //Vec::new();


        for detection in face_boxes.outer_iter() {
            let x_min = detection[0];
            let y_min = detection[1];
            let x_max = detection[2];
            let y_max = detection[3];
            let area = (x_max - x_min) * (x_max - x_min);
            let box_center_width = (x_min + x_max) / 2.0;
            let box_center_height = (y_min + y_max) / 2.0;
            if (box_center_width >= margin_edge)
                && (box_center_width <= img_shape.width as f32 - margin_edge)
                && (box_center_height >= margin_edge)
                && (box_center_height <= img_shape.height as f32 - margin_edge)
                && (area / (img_shape.height as f32 * img_shape.width as f32) >= self.minimum_face_ratio)
            {
                valid_boxes.push(detection.to_vec());
            }
        }

        let mut center_boxes: Vec<Vec<f32>> = Vec::with_capacity(1); //Vec::new();
        for result in valid_boxes.iter() {
            let box_center_width = (result[0] + result[2]) / 2.0;
            if -margin_center_left <= box_center_width - x_cen && box_center_width - x_cen <= margin_center_right {
                center_boxes.push(result.clone());
            }
        }

        if center_boxes.len() == 0 {
            if valid_boxes.len() == 0 {
                center_boxes =  face_boxes.outer_iter().map(|row| row.to_vec()).collect()
            } else {
                center_boxes = valid_boxes
            }
        }

        let mut outbboxes: Option<Array1<f32>> = None;
        let mut max_size: f32 = 0.0;

        for result in center_boxes.iter() {
            let tem_size = (result[2] - result[0]) + (result[3] - result[1]);
            if tem_size > max_size {
                max_size = tem_size;
                outbboxes = Some(<Array1<f32>>::from(result.to_owned()));
            }
        }
        drop(center_boxes);

        if outbboxes.is_none() {
            return Ok((None, None))
        }

        let mut out_keypoint: Option<Array2<f32>> = None;
        if let Some(kps) = key_points {
            for (bbox, key_point) in face_boxes.outer_iter().zip(kps.outer_iter()) {

                if let Some(_outbboxes) = &outbboxes {
                    let (xmin_out, ymin_out, xmax_out, ymax_out, _score_out) = (
                        _outbboxes[0],
                        _outbboxes[1],
                        _outbboxes[2],
                        _outbboxes[3],
                        _outbboxes[4],
                    );
                    let (x, y, x2, y2, _score) = (bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]);
                    if (xmin_out - x).abs() <= 2.0
                        && (ymin_out - y).abs() <= 2.0
                        && (xmax_out - x2).abs() <= 2.0
                        && (ymax_out - y2).abs() <= 2.0
                    {
                        out_keypoint = Some(key_point.to_owned());
                        break;
                    }
                }
            }
        }
        Ok((outbboxes, out_keypoint))
    }
}

#[cfg(test)]
mod tests {
    use crate::pipeline::module::face_detection::RetinaFaceDetection;
    use crate::pipeline::module::face_selection::FaceSelection;
    use crate::pipeline::triton_client::client::triton::ModelConfigRequest;
    use crate::pipeline::triton_client::client::TritonInferenceClient;
    use crate::pipeline::utils::utils::byte_data_to_opencv;

    #[tokio::test]
    async fn test_face_selection() {
    }
}