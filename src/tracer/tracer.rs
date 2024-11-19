use crate::config::settings::SETTINGS;
use opentelemetry::trace::TraceError;
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::trace::{config, Config, TracerProvider};
use opentelemetry_sdk::{runtime, Resource};
use opentelemetry_semantic_conventions::attribute::SERVICE_NAME;

pub fn init_tracer_provider() -> Result<TracerProvider, TraceError> {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&SETTINGS.tracer.uri)
        .build()?;

    Ok(TracerProvider::builder()
        .with_batch_exporter(exporter, runtime::Tokio)
        .with_config(
            Config::default()
                .with_resource(Resource::new(vec![KeyValue::new(SERVICE_NAME, SETTINGS.app.name.clone())])),
        )
        .build())
}
