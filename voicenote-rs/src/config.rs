use anyhow::{Context, Result};
use colored::Colorize;
use dialoguer::{Input, Select};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub vault_path: String,
    pub save_folder: String,
    pub whisper_model: String,
}

pub fn get_config_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("ホームディレクトリが見つかりません")?;
    let config_dir = home.join(".config").join("voicenote");
    fs::create_dir_all(&config_dir)?;
    Ok(config_dir)
}

pub fn load_config(path: &Path) -> Result<Config> {
    let content = fs::read_to_string(path).context("設定ファイルの読み込みに失敗しました")?;
    let config: Config = serde_json::from_str(&content).context("設定ファイルのパースに失敗しました")?;
    Ok(config)
}

pub fn save_config(path: &Path, config: &Config) -> Result<()> {
    let content = serde_json::to_string_pretty(config)?;
    fs::write(path, content)?;
    println!("{} {}", "設定を保存しました:".green(), path.display());
    Ok(())
}

pub fn configure_interactive() -> Result<Config> {
    println!("{}", "========================================".cyan());
    println!("{}", "初回設定".bold().cyan());
    println!("{}", "設定項目を入力してください。".cyan());
    println!("{}", "========================================".cyan());

    let vault_path: String = loop {
        let input: String = Input::new()
            .with_prompt("Obsidian Vaultの絶対パス")
            .interact_text()?;

        let path = PathBuf::from(shellexpand::tilde(&input).to_string());

        if path.exists() && path.is_dir() {
            println!("{} {}", "Vaultパスを確認しました:".green(), path.display());
            break path.to_string_lossy().to_string();
        } else {
            println!(
                "{}",
                "指定されたパスが存在しないか、ディレクトリではありません。".red()
            );
        }
    };

    let save_folder: String = Input::new()
        .with_prompt("保存先フォルダ名（Vault内の相対パス）")
        .default("recordings".to_string())
        .interact_text()?;

    println!("\n{}", "使用するWhisperモデルを選択してください:".bold());
    let models = ["tiny", "base", "small", "medium", "large"];
    let descriptions = [
        "tiny   (最速・精度低)",
        "base   (高速・精度中)",
        "small  (標準)",
        "medium (精度高・時間かかる)",
        "large  (最高精度・最も時間かかる)",
    ];

    let selection = Select::new()
        .with_prompt("選択")
        .items(&descriptions)
        .default(2)
        .interact()?;

    let whisper_model = models[selection].to_string();
    println!("{} '{}'", "モデルを選択しました:".green(), whisper_model);

    Ok(Config {
        vault_path,
        save_folder,
        whisper_model,
    })
}
