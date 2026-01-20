use anyhow::{Context, Result};
use colored::Colorize;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{SampleFormat, WavSpec, WavWriter};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub const SAMPLE_RATE: u32 = 16000;

pub fn record_audio() -> Result<Vec<f32>> {
    let recording_data: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let is_recording = Arc::new(AtomicBool::new(true));

    let is_recording_ctrlc = Arc::clone(&is_recording);
    ctrlc::set_handler(move || {
        is_recording_ctrlc.store(false, Ordering::SeqCst);
        println!("\n{}", "録音を停止しています...".yellow());
    })?;

    println!("{}", "========================================".green());
    println!("{}", "録音を開始します".bold().green());
    println!("{}", "Ctrl+C で録音を終了します".yellow());
    println!("{}", "========================================".green());

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("入力デバイスが見つかりません")?;

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Default,
    };

    let recording_data_clone = Arc::clone(&recording_data);
    let is_recording_stream = Arc::clone(&is_recording);

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            if is_recording_stream.load(Ordering::SeqCst) {
                if let Ok(mut buffer) = recording_data_clone.lock() {
                    buffer.extend_from_slice(data);
                }
            }
        },
        move |err| {
            eprintln!("{} {}", "録音エラー:".red(), err);
        },
        None,
    )?;

    stream.play()?;

    while is_recording.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    drop(stream);

    println!("{}", "録音完了".green());

    let data = recording_data
        .lock()
        .map_err(|_| anyhow::anyhow!("ロックの取得に失敗"))?
        .clone();

    if data.is_empty() {
        anyhow::bail!("録音データがありません。");
    }

    Ok(data)
}

pub fn save_wav(path: &Path, data: &[f32]) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)?;

    for &sample in data {
        let sample_i16 = (sample * 32767.0) as i16;
        writer.write_sample(sample_i16)?;
    }

    writer.finalize()?;
    Ok(())
}
