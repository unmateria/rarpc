use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::attack::AttackConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub rar_file: PathBuf,
    pub attack_config: AttackConfig,
    pub position: u64,
    pub found_password: Option<String>,
    pub started_at: String,
    pub last_updated: String,
}

impl Session {
    pub fn new(rar_file: &Path, config: &AttackConfig) -> Self {
        let now = Utc::now().to_rfc3339();
        Self {
            rar_file: rar_file.to_path_buf(),
            attack_config: config.clone(),
            position: 0,
            found_password: None,
            started_at: now.clone(),
            last_updated: now,
        }
    }

    pub fn save(&mut self, path: &Path) -> Result<()> {
        self.last_updated = Utc::now().to_rfc3339();
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
}
