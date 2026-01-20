use anyhow::{Context, Result};
use colored::Colorize;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use hound::{WavSpec, WavWriter};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub struct RecordingResult {
    pub data: Vec<f32>,
    pub sample_rate: u32,
}

pub fn record_audio() -> Result<RecordingResult> {
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

    let supported_config = device
        .default_input_config()
        .context("デバイスのデフォルト設定を取得できません")?;

    let actual_sample_rate = supported_config.sample_rate().0;
    let channels = supported_config.channels();

    let config = cpal::StreamConfig {
        channels,
        sample_rate: supported_config.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    let recording_data_clone = Arc::clone(&recording_data);
    let is_recording_stream = Arc::clone(&is_recording);

    let stream = match supported_config.sample_format() {
        SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if is_recording_stream.load(Ordering::SeqCst) {
                    if let Ok(mut buffer) = recording_data_clone.lock() {
                        if channels == 1 {
                            buffer.extend_from_slice(data);
                        } else {
                            for chunk in data.chunks(channels as usize) {
                                buffer.push(chunk[0]);
                            }
                        }
                    }
                }
            },
            move |err| {
                eprintln!("{} {}", "録音エラー:".red(), err);
            },
            None,
        )?,
        SampleFormat::I16 => {
            let recording_data_clone = Arc::clone(&recording_data);
            let is_recording_stream = Arc::clone(&is_recording);
            device.build_input_stream(
                &config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    if is_recording_stream.load(Ordering::SeqCst) {
                        if let Ok(mut buffer) = recording_data_clone.lock() {
                            for chunk in data.chunks(channels as usize) {
                                buffer.push(chunk[0] as f32 / 32768.0);
                            }
                        }
                    }
                },
                move |err| {
                    eprintln!("{} {}", "録音エラー:".red(), err);
                },
                None,
            )?
        }
        SampleFormat::I32 => {
            let recording_data_clone = Arc::clone(&recording_data);
            let is_recording_stream = Arc::clone(&is_recording);
            device.build_input_stream(
                &config,
                move |data: &[i32], _: &cpal::InputCallbackInfo| {
                    if is_recording_stream.load(Ordering::SeqCst) {
                        if let Ok(mut buffer) = recording_data_clone.lock() {
                            for chunk in data.chunks(channels as usize) {
                                buffer.push(chunk[0] as f32 / 2147483648.0);
                            }
                        }
                    }
                },
                move |err| {
                    eprintln!("{} {}", "録音エラー:".red(), err);
                },
                None,
            )?
        }
        format => anyhow::bail!("サポートされていないサンプルフォーマット: {:?}", format),
    };

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

    Ok(RecordingResult {
        data,
        sample_rate: actual_sample_rate,
    })
}

pub fn save_wav(path: &Path, data: &[f32], sample_rate: u32) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)?;

    for &sample in data {
        let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(sample_i16)?;
    }

    writer.finalize()?;
    Ok(())
}
