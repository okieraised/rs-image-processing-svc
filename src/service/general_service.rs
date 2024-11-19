use std::sync::Arc;
use anyhow::Error;
use log::error;
use crate::models::general_model::{GeneralExtractionInput, GeneralExtractionResultOutput};
use crate::pipeline::general_pipeline::general_pipeline::{GeneralPipeline};

#[derive(Clone)]
pub struct GeneralService {
    general_pipeline: Arc<GeneralPipeline>
}

impl GeneralService {
    pub fn new(general_pipeline: &Arc<GeneralPipeline>) -> Self {
        GeneralService {
            general_pipeline: Arc::clone(general_pipeline),
        }
    }

    pub async fn extract_general_image(&self, input: GeneralExtractionInput) ->  Result<GeneralExtractionResultOutput, Error> {

        let result = match self.general_pipeline.extract(&input.im_bytes.to_owned(), input.is_enroll.to_owned()).await {
            Ok(result) => {result}
            Err(e) => {
                error!("failed to extract face: {e}");
                return Err(e)
            }
        };

        drop(input.im_bytes);

        let mut facial_feature: Option<Vec<f32>> = Some(Vec::with_capacity(512));
        let face_count = result.face_count;
        let quality_score = result.quality_score;
        let face_quality = result.face_quality;

        if let Some(_feature) = result.facial_feature {
            facial_feature = Some(_feature.to_vec());
        }


        Ok(GeneralExtractionResultOutput {
            face_count,
            face_quality,
            quality_score,
            facial_feature,
        })
    }
}