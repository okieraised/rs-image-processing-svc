use anyhow::Error;
use ndarray::{Array1};
use serde::{Deserialize, Serialize};
use crate::pipeline::model_config::config::{FaceAlignmentConfig, FaceDetectionConfig, FaceIdentificationConfig, FaceQualityClass, FaceQualityConfig, FaceSelectionConfig, match_face_quality};
use crate::pipeline::module::face_alignment::FaceAlignment;
use crate::pipeline::module::face_detection::RetinaFaceDetection;
use crate::pipeline::module::face_extraction::FaceExtraction;
use crate::pipeline::module::face_quality::FaceQuality;
use crate::pipeline::module::face_selection::FaceSelection;
use crate::pipeline::triton_client::client::triton::ModelConfigRequest;
use crate::pipeline::triton_client::client::TritonInferenceClient;
use crate::pipeline::utils::utils::byte_data_to_opencv;

#[derive(Clone)]
pub struct GeneralPipeline {
    face_detection: RetinaFaceDetection,
    face_selection: FaceSelection,
    face_alignment: FaceAlignment,
    face_quality: FaceQuality,
    face_extraction: FaceExtraction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralFaceExtractionResult {
    pub face_count: i32,
    pub face_quality: Option<FaceQualityClass>,
    pub quality_score: Option<f32>,
    pub facial_feature: Option<Array1<f32>>,
}

impl GeneralFaceExtractionResult {
    fn new() -> GeneralFaceExtractionResult {
        GeneralFaceExtractionResult {
            face_count: 0,
            face_quality: None,
            quality_score: None,
            facial_feature: Some(Array1::<f32>::default(512)),
        }
    }
}

impl GeneralPipeline {
    pub async fn new(
        triton_host: &str,
        triton_port: &str,
    ) -> Result<Self, Error> {

        // Init model config
        let face_detection_cfg = FaceDetectionConfig::new();
        let face_selection_cfg = FaceSelectionConfig::new();
        let face_align_cfg = FaceAlignmentConfig::new();
        let face_quality_cfg = FaceQualityConfig::new();
        let face_extraction_cfg = FaceIdentificationConfig::new();

        // Init triton client
        let triton_infer_client = match TritonInferenceClient::new(triton_host, triton_port).await {
            Ok(triton_infer_client) => triton_infer_client,
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        // Query face detection model config
        let face_detection_model_config = match triton_infer_client
            .model_config(ModelConfigRequest {
                name: face_detection_cfg.model_name.to_string(),
                version: "".to_string(),
            }).await {
            Ok(model_config_resp) => {model_config_resp}
            Err(e) => return Err(Error::from(e))
        };

        // Query face quality model config
        let face_quality_model_config = match triton_infer_client
            .model_config(ModelConfigRequest {
                name: face_quality_cfg.model_name.to_string(),
                version: "".to_string(),
            }).await {
            Ok(model_config_resp) => {model_config_resp}
            Err(e) => return Err(Error::from(e))
        };

        // Query face extraction model config
        let face_extraction_model_config = match triton_infer_client
            .model_config(ModelConfigRequest {
                name: face_extraction_cfg.model_name.to_string(),
                version: "".to_string(),
            }).await {
            Ok(model_config_resp) => {model_config_resp}
            Err(e) => return Err(Error::from(e))
        };


        // face detection model
        let face_detection = match RetinaFaceDetection::new(
            triton_infer_client.clone(),
            face_detection_model_config,
            face_detection_cfg.model_name,
            face_detection_cfg.image_size,
            face_detection_cfg.max_batch_size,
            face_detection_cfg.confidence_threshold,
            face_detection_cfg.iou_threshold,
        ).await {
            Ok(face_detection) => {face_detection}
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        // face selection model
        let face_selection = FaceSelection::new(
            face_selection_cfg.margin_center_left_ratio,
            face_selection_cfg.margin_center_right_ratio,
            face_selection_cfg.margin_edge_ratio,
            face_selection_cfg.minimum_face_ratio,
        ).await;

        // Face alignment model
        let face_alignment = FaceAlignment::new(
            face_align_cfg.image_size,
            face_align_cfg.standard_landmarks
        );

        // Face quality model
        let face_quality = match FaceQuality::new(
            triton_infer_client.clone(),
            face_quality_model_config,
            face_quality_cfg.model_name,
            face_quality_cfg.image_size,
            face_quality_cfg.threshold,
        ).await {
            Ok(face_quality) => {face_quality}
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        // Face extraction model
        let face_extraction = match FaceExtraction::new(
            triton_infer_client.clone(),
            face_extraction_model_config,
            face_extraction_cfg.model_name,
            face_extraction_cfg.image_size,
            face_extraction_cfg.batch_size,
        ).await {
            Ok(face_extraction) => {face_extraction}
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        Ok(GeneralPipeline {
            face_detection,
            face_selection,
            face_alignment,
            face_quality,
            face_extraction,
        })
    }

    pub async fn extract(&self, im_bytes: &[u8], is_enroll: Option<bool>) -> Result<GeneralFaceExtractionResult, Error> {
        let enroll = is_enroll.unwrap_or(false);

        let mut general_extraction_result = GeneralFaceExtractionResult::new();
        let image = match byte_data_to_opencv(im_bytes) {
            Ok(image) => {image}
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        let (detections, key_points)  = match self.face_detection.call(image.clone()).await {
            Ok((detections, key_points)) => {(detections, key_points)}
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        let face_count = detections.dim().0 as i32;
        general_extraction_result.face_count = face_count;

        let (selected_face_box, selected_face_point) = match self.face_selection.call(&image, detections, key_points, Some(enroll)) {
            Ok((selected_face_box, selected_face_point)) => {(selected_face_box, selected_face_point)}
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        if selected_face_box.is_some() {
            let aligned_face_image = match self.face_alignment.call(&image, selected_face_box.clone(), selected_face_point) {
                Ok(aligned_face_image) => {aligned_face_image}
                Err(e) => {
                    return Err(Error::from(e))
                }
            };

            let (quality_score, quality_class) = match self.face_quality.call(aligned_face_image.clone()).await {
                Ok((quality_score, quality_class)) => {(quality_score, quality_class)}
                Err(e) => {
                    return Err(Error::from(e))
                }
            };

            let facial_feature = match self.face_extraction.call(aligned_face_image).await {
                Ok(facial_feature) => {facial_feature}
                Err(e) => {
                    return Err(Error::from(e))
                }
            };
            general_extraction_result.facial_feature = Some(facial_feature[0].to_owned().into_shape((facial_feature[0].len(),)).unwrap());
            general_extraction_result.face_count = face_count;
            general_extraction_result.face_quality = Some(match_face_quality(quality_class[0].to_owned()));
            general_extraction_result.quality_score = Some(quality_score[0]);
            drop(facial_feature);
        }


        Ok(general_extraction_result)
    }
}


#[cfg(test)]
mod tests {
    use crate::pipeline::general_pipeline::general_pipeline::GeneralPipeline;
    use crate::pipeline::triton_client::client::TritonInferenceClient;

    #[tokio::test]
    async fn test_pipeline() {

    }
}
