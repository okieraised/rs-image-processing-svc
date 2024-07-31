use serde::Serialize;
use crate::config::settings::{Logger, SETTINGS, Settings};

#[derive(Serialize)]
pub struct LoggerExtraFields {
    pub request_id: String,
}

pub fn setup_logger() {
    let mut log_level: log::LevelFilter = log::LevelFilter::Debug;

    let setting_level = &SETTINGS.logger.clone().unwrap_or(Logger { level: "info".to_string()}).level;

    match setting_level.as_str() {
        "trace" => {
            log_level = log::LevelFilter::Trace;
        }
        "debug" => {
            log_level = log::LevelFilter::Debug;
        }
        "warn" => {
            log_level = log::LevelFilter::Warn;
        }
        "error" => {
            log_level = log::LevelFilter::Error;
        }
        _ => {
            log_level = log::LevelFilter::Info;
        }

    };

    env_logger::builder()
        .filter_level(log_level)
        .format_timestamp_micros()
        .format(ecs_logger::format)
        .target(env_logger::Target::Stdout)
        .target(env_logger::Target::Stderr)
        .init();
}


#[cfg(test)]
mod tests {
    use log::{error, info};
    use super::*;

    #[test]
    fn test_logger() {
        setup_logger();
        info!("test log info");
    }
}