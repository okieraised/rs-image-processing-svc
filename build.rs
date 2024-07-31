fn main() {
    tonic_build::compile_protos("triton_proto/model_config.proto").unwrap();
    tonic_build::compile_protos("triton_proto/grpc_service.proto").unwrap();
}