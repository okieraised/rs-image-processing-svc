use std::sync::Arc;
use crate::pipeline::antispoofing_pipeline::antispoofing_pipeline::AntiSpoofingPipeline;
use crate::service::antispoofing_service::AntiSpoofingService;

#[derive(Clone)]
pub struct AntiSpoofingState {
    pub anti_spoofing_service: AntiSpoofingService,
}

impl AntiSpoofingState {
    pub fn new(pipeline: &Arc<AntiSpoofingPipeline>) -> Self {
        Self {
            anti_spoofing_service: AntiSpoofingService::new(pipeline),
        }
    }
}