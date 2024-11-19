use std::sync::Arc;
use anyhow::Error;
use log::error;
use crate::models::general_model::{GeneralExtractionInput, GeneralExtractionResultOutput};
use crate::models::antispoofing_model::{AntiSpoofingExtractionInput, AntiSpoofingExtractionResultOutput};
use crate::pipeline::antispoofing_pipeline::antispoofing_pipeline::AntiSpoofingPipeline;

#[derive(Clone)]
pub struct AntiSpoofingService {
    antispoofing_pipeline: Arc<AntiSpoofingPipeline>
}

impl AntiSpoofingService {
    pub fn new(antispoofing_pipeline: &Arc<AntiSpoofingPipeline>) -> Self {
        AntiSpoofingService {
            antispoofing_pipeline: Arc::clone(antispoofing_pipeline),
        }
    }

    pub async fn extract_antispoofing_image(&self, input: AntiSpoofingExtractionInput) ->  Result<AntiSpoofingExtractionResultOutput, Error> {

        let result = match self.antispoofing_pipeline.extract(&input.im_bytes.to_owned(), input.spoofing_check.to_owned(), input.is_enroll.to_owned()).await {
            Ok(result) => {result}
            Err(e) => {
                error!("failed to extract face: {e}");
                return Err(e)
            }
        };

        let mut facial_feature: Option<Vec<f32>> = None;
        let face_count = result.face_count;
        let spoofing_check = result.spoofing_check;
        let face_quality = result.face_quality;

        if let Some(_feature) = result.facial_feature {
            facial_feature = Some(_feature.to_vec());
        }

        Ok(AntiSpoofingExtractionResultOutput {
            face_count,
            face_quality,
            facial_feature,
            spoofing_check,
        })
    }


}