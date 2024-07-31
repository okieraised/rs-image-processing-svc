use anyhow::Error;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use crate::pipeline::model_config::config::{FaceAlignmentConfig, FaceAntiSpoofingClass, FaceAntiSpoofingConfig, FaceDetectionConfig, FaceIdentificationConfig, FaceQualityAssessmentConfig, FaceQualityClass, FaceQualityConfig, FaceSelectionConfig, match_face_anti_spoofing, match_face_quality};
use crate::pipeline::module::face_alignment::FaceAlignment;
use crate::pipeline::module::face_antispoofing::FaceAntiSpoofing;
use crate::pipeline::module::face_detection::RetinaFaceDetection;
use crate::pipeline::module::face_extraction::FaceExtraction;
use crate::pipeline::module::face_quality::FaceQuality;
use crate::pipeline::module::face_quality_assessment::FaceQualityAssessment;
use crate::pipeline::module::face_selection::FaceSelection;
use crate::pipeline::triton_client::client::triton::{ModelConfigRequest, ModelConfigResponse};
use crate::pipeline::triton_client::client::TritonInferenceClient;
use crate::pipeline::utils::utils::byte_data_to_opencv;

#[derive(Clone)]
pub struct AntiSpoofingPipeline {
    face_detection: RetinaFaceDetection,
    face_selection: FaceSelection,
    face_alignment: FaceAlignment,
    face_quality: FaceQuality,
    face_quality_assessment: FaceQualityAssessment,
    face_anti_spoofing: FaceAntiSpoofing,
    face_extraction: FaceExtraction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntiSpoofingFaceExtractionResult {
    pub face_count:i32,
    pub facial_feature: Option<Array1<f32>>,
    pub face_quality: Option<FaceQualityClass>,
    pub spoofing_check: Option<FaceAntiSpoofingClass>,
}

impl AntiSpoofingFaceExtractionResult {
    fn new() -> AntiSpoofingFaceExtractionResult {
        AntiSpoofingFaceExtractionResult {
            face_count: 0,
            facial_feature: None,
            face_quality: Some(FaceQualityClass::Good),
            spoofing_check: Some(FaceAntiSpoofingClass::Real),
        }
    }
}

impl AntiSpoofingPipeline {
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
        let face_anti_spoofing_cfg = FaceAntiSpoofingConfig::new();
        let face_quality_assessment_cfg = FaceQualityAssessmentConfig::new();

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

        // Query face anti-spoofing model config
        let mut antispoofing_model_config: Vec<ModelConfigResponse> = vec![];
        for model_name in &face_anti_spoofing_cfg.model_name {
            let face_antispoofing_model_config = match triton_infer_client
                .model_config(ModelConfigRequest {
                    name: model_name.to_owned(),
                    version: "".to_string(),
                }).await {
                Ok(model_config_resp) => {model_config_resp}
                Err(e) => {
                    return Err(Error::from(e))
                }
            };
            antispoofing_model_config.push(face_antispoofing_model_config)
        }

        // Query face quality assessment model config
        let face_quality_assessment_model_config = match triton_infer_client
            .model_config(ModelConfigRequest {
                name: face_quality_assessment_cfg.model_name.to_string(),
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

        // Face anti-spoofing model
        let face_anti_spoofing = match FaceAntiSpoofing::new(
            triton_infer_client.clone(),
            antispoofing_model_config,
            face_anti_spoofing_cfg.model_name,
            face_anti_spoofing_cfg.image_size,
            face_anti_spoofing_cfg.scale,
            face_anti_spoofing_cfg.batch_size,
            face_anti_spoofing_cfg.threshold,
        ).await {
            Ok(face_anti_spoofing) => {face_anti_spoofing}
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        // Face quality assessment model
        let face_quality_assessment = match FaceQualityAssessment::new(
            triton_infer_client.clone(),
            face_quality_assessment_model_config,
            face_quality_assessment_cfg.model_name,
            face_quality_assessment_cfg.image_size,
            face_quality_assessment_cfg.batch_size,
            face_quality_assessment_cfg.threshold,
        ).await {
            Ok(face_quality_assessment) => {face_quality_assessment}
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        Ok(AntiSpoofingPipeline {
            face_detection,
            face_selection,
            face_alignment,
            face_quality,
            face_quality_assessment,
            face_anti_spoofing,
            face_extraction,
        })
    }

    pub async fn extract(&self, im_bytes: &[u8], is_spoofing_check: Option<bool>, is_enroll: Option<bool>) -> Result<AntiSpoofingFaceExtractionResult, Error> {

        let mut antispoofing_extraction_result = AntiSpoofingFaceExtractionResult::new();

        let spoofing_check = is_spoofing_check.unwrap_or(false);
        let enroll = is_enroll.unwrap_or(false);

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
        antispoofing_extraction_result.face_count = face_count;

        let (selected_face_box, selected_face_point) = match self.face_selection.call(&image.clone(), detections, key_points, is_enroll) {
            Ok((selected_face_box, selected_face_point)) => {(selected_face_box, selected_face_point)}
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        if let Some(_selected_face_box) = selected_face_box {
            if spoofing_check {
                let model_spoofing_result = match self.face_anti_spoofing.call(image.clone(), _selected_face_box.clone()).await {
                    Ok(model_spoofing_result) => {model_spoofing_result}
                    Err(e) => {
                        return Err(Error::from(e))
                    }
                };
                antispoofing_extraction_result.spoofing_check = Some(match_face_anti_spoofing(model_spoofing_result[0].to_vec()[0] as usize));
            }


            let aligned_face_image = match self.face_alignment.call(&image, Some(_selected_face_box.clone()), selected_face_point) {
                Ok(aligned_face_image) => {aligned_face_image}
                Err(e) => {
                    return Err(Error::from(e))
                }
            };

            let aligned_img_arr = aligned_face_image;

            let (quality_score, quality_class) = match self.face_quality.call(aligned_img_arr.clone()).await {
                Ok((quality_score, quality_class)) => {(quality_score, quality_class)}
                Err(e) => {
                    return Err(Error::from(e))
                }
            };

            // face quality assessment
            let (quality_assessment_score, quality_assessment_class) = match self.face_quality_assessment.call(aligned_img_arr.clone()).await {
                Ok((quality_assessment_score, quality_assessment_class)) => {(quality_assessment_score[0], quality_assessment_class[0])}
                Err(e) => {
                    return Err(Error::from(e))
                }
            };

            if !enroll {
                if match_face_quality(quality_class[0].to_owned()) == FaceQualityClass::WearingMask {
                    antispoofing_extraction_result.face_quality = Some(match_face_quality(quality_class[0].to_owned()));
                    return Ok(antispoofing_extraction_result)
                } else {
                    let facial_feature = match self.face_extraction.call(aligned_img_arr.clone()).await {
                        Ok(facial_feature) => {facial_feature}
                        Err(e) => {
                            return Err(Error::from(e))
                        }
                    };
                    antispoofing_extraction_result.facial_feature = Some(facial_feature[0].to_owned().into_shape((facial_feature[0].len(),)).unwrap());
                }
            } else {
                if match_face_quality(quality_class[0].to_owned()) == FaceQualityClass::Good && match_face_quality(quality_assessment_class.to_owned() as usize) == FaceQualityClass::Good {
                    let facial_feature = match self.face_extraction.call(aligned_img_arr.clone()).await {
                        Ok(facial_feature) => {facial_feature}
                        Err(e) => {
                            return Err(Error::from(e))
                        }
                    };
                    antispoofing_extraction_result.facial_feature = Some(facial_feature[0].to_owned().into_shape((facial_feature[0].len(),)).unwrap());
                } else {
                    antispoofing_extraction_result.face_quality = Some(FaceQualityClass::Bad)
                }
            }

        }
        Ok(antispoofing_extraction_result)
    }
}