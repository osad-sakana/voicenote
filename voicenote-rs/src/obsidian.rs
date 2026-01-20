use anyhow::Result;
use chrono::Local;
use colored::Colorize;
use std::fs;
use std::path::{Path, PathBuf};

pub fn save_to_obsidian(
    vault_path: &Path,
    save_folder: &str,
    transcription: &str,
) -> Result<PathBuf> {
    let save_dir = vault_path.join(save_folder);
    fs::create_dir_all(&save_dir)?;

    let now = Local::now();
    let timestamp = now.format("%Y-%m-%d_%H%M%S").to_string();
    let filename = format!("{}_raw.md", timestamp);
    let filepath = save_dir.join(&filename);

    let iso_timestamp = now.to_rfc3339();
    let content = format!(
        r#"---
created: {}
type: transcription
tags:
  - recording
  - raw
---
{}"#,
        iso_timestamp, transcription
    );

    fs::write(&filepath, content)?;
    println!("{}", "保存完了".green());

    Ok(filepath)
}
