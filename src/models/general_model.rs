use bytes::Bytes;
use serde::{Deserialize, Serialize};
use crate::pipeline::model_config::config::FaceQualityClass;


#[derive(Clone, Serialize)]
pub struct GeneralExtractionResultOutput {
    pub face_count: i32,
    pub face_quality: Option<FaceQualityClass>,
    pub quality_score: Option<f32>,
    pub facial_feature: Option<Vec<f32>>,
}

impl Default for GeneralExtractionResultOutput {
    fn default() -> Self {
        GeneralExtractionResultOutput {
            face_count: 0,
            face_quality: None,
            quality_score: None,
            facial_feature: None,
        }
    }
}

#[derive(Clone)]
pub struct GeneralExtractionInput {
    pub im_bytes: Bytes,
    pub is_enroll: Option<bool>,
}