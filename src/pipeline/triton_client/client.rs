use anyhow::{Error, Result};
use tonic::transport::Channel;

pub mod triton {
    tonic::include_proto!("inference");
}

use triton::grpc_inference_service_client::GrpcInferenceServiceClient;
use triton::{ModelInferRequest, ModelReadyRequest, ModelConfigRequest};
use crate::pipeline::triton_client::client::triton::{CudaSharedMemoryRegisterRequest, CudaSharedMemoryRegisterResponse,
                                           CudaSharedMemoryStatusRequest, CudaSharedMemoryStatusResponse,
                                           CudaSharedMemoryUnregisterRequest, CudaSharedMemoryUnregisterResponse,
                                           ModelConfigResponse, ModelInferResponse,
                                           ModelMetadataRequest, ModelMetadataResponse,
                                           ModelReadyResponse, ModelStatisticsRequest,
                                           ModelStatisticsResponse, RepositoryIndexRequest,
                                           RepositoryIndexResponse, RepositoryModelLoadRequest,
                                           RepositoryModelLoadResponse, RepositoryModelUnloadRequest,
                                           RepositoryModelUnloadResponse, ServerLiveRequest,
                                           ServerLiveResponse, ServerMetadataRequest,
                                           ServerMetadataResponse, ServerReadyRequest,
                                           ServerReadyResponse, SystemSharedMemoryRegisterRequest,
                                           SystemSharedMemoryRegisterResponse, SystemSharedMemoryStatusRequest,
                                           SystemSharedMemoryStatusResponse, SystemSharedMemoryUnregisterRequest,
                                           SystemSharedMemoryUnregisterResponse, TraceSettingRequest,
                                           TraceSettingResponse};

#[derive(Debug, Clone)]
pub struct TritonInferenceClient {
    c: GrpcInferenceServiceClient<Channel>,
}

macro_rules! wrap_method_with_args {
    ($doc:literal, $name:ident, $req_type:ty, $resp_type:ty) => {
        #[doc=$doc]
        pub async fn $name(&self, req: $req_type) -> Result<$resp_type, Error> {
            let response = self.c.clone().$name(tonic::Request::new(req)).await?;
            Ok(response.into_inner())
        }
    };
}

macro_rules! wrap_method_no_args {
    ($doc:literal, $name:ident, $req_type:ty, $resp_type:ty) => {
        #[doc=$doc]
        pub async fn $name(&self) -> Result<$resp_type, Error> {
            let req: $req_type = Default::default();
            let response = self.c.clone().$name(tonic::Request::new(req)).await?;
            Ok(response.into_inner())
        }
    };
}

impl TritonInferenceClient {
    pub(crate) async fn new(host: &str, port: &str) -> Result<Self, Error> {
        let channel_url = format!("{}:{}", host, port);
        let channel = match Channel::from_shared(channel_url).expect("url must be valid")
            .connect()
            .await {
          Ok(channel) => channel,
            Err(e) => return Err(Error::from(e))
        };

        let client = GrpcInferenceServiceClient::new(channel);

        Ok(TritonInferenceClient {
            c: client,
        })
    }

    wrap_method_no_args!(
        "Check liveness of the inference server.",
        server_live,
        ServerLiveRequest,
        ServerLiveResponse
    );

    wrap_method_no_args!(
        "Check readiness of the inference server.",
        server_ready,
        ServerReadyRequest,
        ServerReadyResponse
    );

    wrap_method_with_args!(
        "Check readiness of a model in the inference server.",
        model_ready,
        ModelReadyRequest,
        ModelReadyResponse
    );

    wrap_method_no_args!(
        "Get server metadata.",
        server_metadata,
        ServerMetadataRequest,
        ServerMetadataResponse
    );

    wrap_method_with_args!(
        "Get model metadata.",
        model_metadata,
        ModelMetadataRequest,
        ModelMetadataResponse
    );

    wrap_method_with_args!(
        "Perform inference using specific model.",
        model_infer,
        ModelInferRequest,
        ModelInferResponse
    );

    wrap_method_with_args!(
        "Get model configuration.",
        model_config,
        ModelConfigRequest,
        ModelConfigResponse
    );

    wrap_method_with_args!(
        "Get the cumulative inference statistics for a model.",
        model_statistics,
        ModelStatisticsRequest,
        ModelStatisticsResponse
    );

    wrap_method_with_args!(
        "Get the index of model repository contents.",
        repository_index,
        RepositoryIndexRequest,
        RepositoryIndexResponse
    );

    wrap_method_with_args!(
        "Load or reload a model from a repository.",
        repository_model_load,
        RepositoryModelLoadRequest,
        RepositoryModelLoadResponse
    );

    wrap_method_with_args!(
        "Unload a model.",
        repository_model_unload,
        RepositoryModelUnloadRequest,
        RepositoryModelUnloadResponse
    );

    wrap_method_with_args!(
        "Get the status of all registered system-shared-memory regions.",
        system_shared_memory_status,
        SystemSharedMemoryStatusRequest,
        SystemSharedMemoryStatusResponse
    );

    wrap_method_with_args!(
        "Register a system-shared-memory region.",
        system_shared_memory_register,
        SystemSharedMemoryRegisterRequest,
        SystemSharedMemoryRegisterResponse
    );

    wrap_method_with_args!(
        "Unregister a system-shared-memory region.",
        system_shared_memory_unregister,
        SystemSharedMemoryUnregisterRequest,
        SystemSharedMemoryUnregisterResponse
    );

    wrap_method_with_args!(
        "Get the status of all registered CUDA-shared-memory regions.",
        cuda_shared_memory_status,
        CudaSharedMemoryStatusRequest,
        CudaSharedMemoryStatusResponse
    );

    wrap_method_with_args!(
        "Register a CUDA-shared-memory region.",
        cuda_shared_memory_register,
        CudaSharedMemoryRegisterRequest,
        CudaSharedMemoryRegisterResponse
    );

    wrap_method_with_args!(
        "Unregister a CUDA-shared-memory region.",
        cuda_shared_memory_unregister,
        CudaSharedMemoryUnregisterRequest,
        CudaSharedMemoryUnregisterResponse
    );

    wrap_method_with_args!(
        "Update and get the trace setting of the Triton server.",
        trace_setting,
        TraceSettingRequest,
        TraceSettingResponse
    );

}

#[cfg(test)]
mod tests {
    use crate::triton_client::client::{TritonInferenceClient, RepositoryIndexRequest};
    use crate::triton_client::client::triton::{InferTensorContents, ModelConfigRequest};
    use crate::triton_client::client::triton::ModelInferRequest;
    use crate::triton_client::client::triton::model_infer_request::InferInputTensor;

    #[tokio::test]
    async fn test_repository_index() {
        let client = TritonInferenceClient::new("", "").await.unwrap();

        let models = client
            .repository_index(RepositoryIndexRequest {
                repository_name: "".into(),
                ready: false,
            }).await.unwrap();

        for model in models.models.iter() {
            println!("{:?}", model);
        }
    }

    #[tokio::test]
    async fn test_model_config() {
        let client = TritonInferenceClient::new("", "").await.unwrap();

        let models = client
            .model_config(ModelConfigRequest {
                name: "face_detection_retina".to_string(),
                version: "".to_string(),
            }).await.unwrap();

        let cfg_all = models.config.unwrap();

        for cfg in cfg_all.output.iter() {
            println!("{:?}", cfg);
        }


    }

    #[tokio::test]
    async fn test_model_infer() {
        let client = TritonInferenceClient::new("", "").await.unwrap();

        let req = ModelInferRequest {
            model_name: "face_detection_retina".to_string(),
            model_version: "".to_string(),
            id: "".to_string(),
            parameters: Default::default(),
            inputs: vec![InferInputTensor {
                name: "data".to_string(),
                datatype: "".to_string(),
                shape: vec![1, 3, 640, 640],
                parameters: Default::default(),
                contents: Option::from(InferTensorContents {
                    bool_contents: vec![],
                    int_contents: vec![],
                    int64_contents: vec![],
                    uint_contents: vec![],
                    uint64_contents: vec![],
                    fp32_contents: vec![],
                    fp64_contents: vec![],
                    bytes_contents: vec![],
                }),
            }],
            outputs: vec![],
            raw_input_contents: vec![],
        };

        let models = client
            .model_infer(req).await.unwrap();

        println!("{:?}", models.raw_output_contents);
    }
}
