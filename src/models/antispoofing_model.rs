use bytes::Bytes;
use serde::{Deserialize, Serialize};
use crate::pipeline::model_config::config::{FaceAntiSpoofingClass, FaceQualityClass};


#[derive(Clone, Serialize, Deserialize)]
pub struct AntiSpoofingExtractionResultOutput {
    pub face_count: i32,
    pub face_quality: Option<FaceQualityClass>,
    pub spoofing_check: Option<FaceAntiSpoofingClass>,
    pub facial_feature: Option<Vec<f32>>,
}

impl Default for AntiSpoofingExtractionResultOutput {
    fn default() -> Self {
        AntiSpoofingExtractionResultOutput {
            face_count: 0,
            face_quality: None,
            spoofing_check: None,
            facial_feature: None,
        }
    }
}

#[derive(Clone)]
pub struct AntiSpoofingExtractionInput {
    pub im_bytes: Bytes,
    pub is_enroll: Option<bool>,
    pub spoofing_check: Option<bool>,
}