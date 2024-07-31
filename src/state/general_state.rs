use std::sync::Arc;
use crate::pipeline::general_pipeline::general_pipeline::GeneralPipeline;
use crate::service::general_service::GeneralService;

#[derive(Clone)]
pub struct GeneralState {
    pub general_service: GeneralService,
}

impl GeneralState {
    pub fn new(pipeline: &Arc<GeneralPipeline>) -> Self {
        Self {
            general_service: GeneralService::new(pipeline),
        }
    }
}