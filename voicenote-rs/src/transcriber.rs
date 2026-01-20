use anyhow::{Context, Result};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{Read, Write};
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

fn download_model(model_name: &str, model_path: &Path) -> Result<()> {
    let model_file = format!("ggml-{}.bin", model_name);
    let url = format!(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
        model_file
    );

    println!(
        "{}",
        format!("モデル '{}' をダウンロード中...", model_name).cyan()
    );
    println!("{}", format!("URL: {}", url).cyan());

    let response = ureq::get(&url).call().context("ダウンロードに失敗しました")?;

    let total_size: u64 = response
        .header("Content-Length")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let pb = if total_size > 0 {
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {bytes} ダウンロード中...")
                .unwrap(),
        );
        pb.enable_steady_tick(std::time::Duration::from_millis(100));
        pb
    };

    let mut reader = response.into_reader();
    let mut file = std::fs::File::create(model_path)?;
    let mut buffer = [0u8; 8192];
    let mut downloaded: u64 = 0;

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        file.write_all(&buffer[..bytes_read])?;
        downloaded += bytes_read as u64;
        pb.set_position(downloaded);
    }

    pb.finish_with_message("ダウンロード完了");
    println!("{}", "モデルのダウンロードが完了しました".green());

    Ok(())
}

fn get_model_path(model_name: &str, config_dir: &Path) -> Result<std::path::PathBuf> {
    let models_dir = config_dir.join("models");
    std::fs::create_dir_all(&models_dir)?;

    let model_file = format!("ggml-{}.bin", model_name);
    let model_path = models_dir.join(&model_file);

    if !model_path.exists() {
        download_model(model_name, &model_path)?;
    }

    Ok(model_path)
}

pub fn transcribe_audio(audio_path: &Path, model_name: &str, config_dir: &Path) -> Result<String> {
    println!(
        "\n{}",
        format!("Whisperモデル '{}' をロード中...", model_name).cyan()
    );

    let model_path = get_model_path(model_name, config_dir)?;

    let spinner_style = ProgressStyle::default_spinner()
        .template("{spinner:.green} {msg}")
        .unwrap();

    let pb = ProgressBar::new_spinner();
    pb.set_style(spinner_style.clone());
    pb.set_message("モデルをロード中...");
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let ctx = WhisperContext::new_with_params(
        model_path.to_str().unwrap(),
        WhisperContextParameters::default(),
    )
    .context("Whisperモデルのロードに失敗しました")?;

    pb.finish_with_message("モデルをロードしました");

    let pb = ProgressBar::new_spinner();
    pb.set_style(spinner_style);
    pb.set_message("文字起こし中...");
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let audio_data = load_wav_as_samples(audio_path)?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("ja"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    let mut state = ctx.create_state().context("ステートの作成に失敗しました")?;
    state
        .full(params, &audio_data)
        .context("文字起こしに失敗しました")?;

    let num_segments = state.full_n_segments().context("セグメント数の取得に失敗")?;
    let mut segments: Vec<String> = Vec::new();

    for i in 0..num_segments {
        if let Ok(text) = state.full_get_segment_text(i) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                segments.push(trimmed.to_string());
            }
        }
    }

    pb.finish_with_message("文字起こし完了");
    println!("{}", "文字起こし完了".green());

    Ok(segments.join("\n\n"))
}

fn load_wav_as_samples(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_value = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_value)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
    };

    if spec.channels == 2 {
        Ok(samples.chunks(2).map(|chunk| chunk[0]).collect())
    } else {
        Ok(samples)
    }
}
