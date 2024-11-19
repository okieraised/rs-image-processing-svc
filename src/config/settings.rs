use config::{Config, ConfigError, Environment, File, FileFormat};
use once_cell::sync::Lazy;
use serde::Deserialize;
use std::{env, fmt};

pub static SETTINGS: Lazy<Settings> = Lazy::new(|| Settings::new().expect("Failed to setup settings"));

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct App {
    pub name: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Server {
    pub http_port: u16,
    pub api_key: String,
    pub request_timeout: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Triton {
    pub faceid_host: String,
    pub faceid_grpc_port: u16,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Logger {
    pub level: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct Tracer {
    pub uri: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct Settings {
    pub environment: Option<String>,
    pub server: Server,
    pub logger: Option<Logger>,
    pub triton: Triton,
    pub tracer: Tracer,
    pub app: App,
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        let run_mode = env::var("RUN_MODE").unwrap_or_else(|_| "development".into());

        let mut builder = Config::builder()
            .add_source(File::with_name("conf/config.toml").format(FileFormat::Toml))
            .add_source(File::with_name("conf/default").required(false))
            .add_source(File::with_name(&format!("config/{run_mode}")).required(false))
            .add_source(File::with_name("conf/local").required(false))
            .add_source(Environment::default().separator("__"));

        if let Ok(port) = env::var("PORT") {
            builder = builder.set_override("server.port", port)?;
        }

        builder.build()?.try_deserialize()
    }
}

impl fmt::Display for Server {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "http://localhost:{}", &self.http_port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let settings = match Settings::new() {
            Ok(settings) => settings,
            Err(e) => {
                println!("{:?}", e);
                return;
            }
        };

        println!("{:?}", settings.logger.unwrap().level)
    }
}
