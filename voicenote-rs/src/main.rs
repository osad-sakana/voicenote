mod config;
mod obsidian;
mod recorder;
mod transcriber;

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use std::path::PathBuf;

use config::{configure_interactive, get_config_dir, load_config, save_config, Config};
use obsidian::save_to_obsidian;
use recorder::record_audio;
use transcriber::transcribe_audio;

#[derive(Parser, Debug)]
#[command(name = "voicenote")]
#[command(about = "Local voice recording and transcription tool for Obsidian")]
struct Args {
    #[arg(long, help = "Run interactive configuration")]
    config: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let config_dir = get_config_dir()?;
    let config_path = config_dir.join("config.json");

    let config: Config = if args.config {
        let cfg = configure_interactive()?;
        save_config(&config_path, &cfg)?;
        cfg
    } else {
        match load_config(&config_path) {
            Ok(cfg) => {
                println!("{}", "設定を読み込みました。".cyan());
                cfg
            }
            Err(_) => {
                let cfg = configure_interactive()?;
                save_config(&config_path, &cfg)?;
                cfg
            }
        }
    };

    let vault_path = PathBuf::from(&config.vault_path);

    let recording = record_audio()?;

    println!("\n{}", "音声データを一時保存中...".cyan());

    let temp_wav = config_dir.join("temp_recording.wav");
    recorder::save_wav(&temp_wav, &recording.data, recording.sample_rate)?;

    let transcription = transcribe_audio(&temp_wav, &config.whisper_model, &config_dir)?;

    std::fs::remove_file(&temp_wav)?;

    println!("\n{}", "Obsidianに保存中...".cyan());
    let saved_path = save_to_obsidian(&vault_path, &config.save_folder, &transcription)?;

    println!("\n{}", "========================================".green());
    println!("{}", "完了!".bold().green());
    println!("{}", "========================================".green());
    println!("{} {}", "保存先:".bold(), saved_path.display());

    Ok(())
}
