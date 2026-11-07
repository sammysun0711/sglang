"""CPU media contracts for MiMo image, audio, and video inputs."""

import io
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from transformers import Qwen2Config, Qwen2Model

from sglang.srt.models.mimo_audio import AudioEncoderMixin
from sglang.srt.multimodal.processors import mimo_v2
from sglang.srt.multimodal.processors.mimo_audio import AudioInput, MiMoAudioPipeline
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def pipeline():
    return MiMoAudioPipeline(
        audio_token_id=151669,
        audio_start_token_id=151673,
        audio_end_token_id=151674,
        audio_channels=20,
    )


def test_image_normalization_matches_flash_checkpoint():
    pixels = torch.tensor([0.0, 127.5, 255.0]).view(1, 3, 1, 1)
    expected = torch.tensor(
        [(0.0 - 0.485) / 0.229, (0.5 - 0.456) / 0.224, (1.0 - 0.406) / 0.225]
    ).view_as(pixels)
    torch.testing.assert_close(
        mimo_v2.MiMoProcessor.standardize_batch(pixels), expected
    )


@pytest.mark.parametrize("height,width", [(20, 28), (900, 1600), (3000, 5000)])
def test_visual_resize_respects_patch_and_pixel_budgets(height, width):
    resized_h, resized_w = mimo_v2.MiMoProcessor.smart_resize(
        height, width, factor=32, min_pixels=8192, max_pixels=262144
    )
    assert resized_h % 32 == resized_w % 32 == 0
    assert 8192 <= resized_h * resized_w <= 262144


def test_frame_sampling_respects_budget():
    count = mimo_v2.smart_nframes(
        {"fps": 1.0, "min_frames": 2, "max_frames": 4},
        total_frames=120,
        video_fps=2,
    )
    assert count == 4


@pytest.mark.parametrize("sample_rate", [16000, 24000, 48000])
def test_audio_waveform_and_encoded_wav_agree(sample_rate):
    if mimo_v2.AudioDecoder is None:
        pytest.skip("A working TorchCodec/FFmpeg installation is required")
    # FLOAT WAV preserves the samples, so both input routes should produce the
    # same mel features even when resampling to the checkpoint's 24 kHz rate.
    samples = (
        np.sin(np.arange(sample_rate) * (2 * np.pi * 440 / sample_rate)) * 0.2
    ).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="FLOAT")
    processor = pipeline()
    decoded, decoded_len = processor.process_audio(AudioInput(buffer.getvalue()))
    direct, direct_len = processor.process_audio(
        AudioInput((torch.from_numpy(samples), sample_rate))
    )
    assert decoded.shape == direct.shape == (101, 128)
    assert decoded_len == direct_len == 7
    assert torch.isfinite(decoded).all()
    torch.testing.assert_close(decoded, direct, atol=2e-4, rtol=2e-4)
    ids = processor.build_audio_placeholder_ids(decoded_len)
    assert ids == [151673] + [151669] * 7 + [151674]


def test_missing_audio_decoder_does_not_silently_drop_video_audio(monkeypatch):
    monkeypatch.setattr(mimo_v2, "AudioDecoder", None)
    with pytest.raises(ValueError, match="torchcodec is required"):
        mimo_v2.MiMoProcessor.has_audio_track(b"video contents")


@pytest.mark.parametrize("full_attention", [False, True])
@torch.inference_mode()
def test_input_local_transformer_respects_its_own_attention_mode(full_attention):
    torch.manual_seed(124)
    config = Qwen2Config(
        vocab_size=8,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        attention_dropout=0.0,
    )
    model = Qwen2Model(config).eval()
    owner = SimpleNamespace(
        input_local_transformer=model, audio_input_full_attention=full_attention
    )
    speech = torch.randn(2, 4, 32)
    changed = speech.clone()
    changed[:, -1] *= -5
    first = AudioEncoderMixin.apply_input_local_transformer(owner, speech)[:, 0]
    second = AudioEncoderMixin.apply_input_local_transformer(owner, changed)[:, 0]
    if full_attention:
        assert (first - second).abs().max() > 1e-4
    else:
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("with_audio", [False, True])
def test_video_frames_and_short_audio_track(tmp_path, with_audio, monkeypatch):
    if mimo_v2.AudioDecoder is None or not shutil.which("ffmpeg"):
        pytest.skip("A working TorchCodec/FFmpeg installation is required")
    path = tmp_path / "video.mp4"
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=64x64:d=3:r=2",
    ]
    if with_audio:
        command += [
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=24000:duration=1",
        ]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if with_audio:
        command += ["-c:a", "aac"]
    command += [str(path)]
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    assert mimo_v2.MiMoProcessor.has_audio_track(str(path)) is with_audio
    assert mimo_v2.MiMoProcessor.has_audio_track(path.read_bytes()) is with_audio
    video = VideoDecoderWrapper(str(path))
    assert len(video) == 6
    frames = video.get_frames_at([0, 5])
    assert frames.shape == (2, 64, 64, 3)
    assert np.mean(frames[..., 0]) > 200
    assert np.mean(frames[..., 1:]) < 10
    # Exercise CPU frame sampling without the wrapper's accelerator-only pinning.
    monkeypatch.setattr(
        video,
        "get_frames_as_tensor",
        lambda indices: torch.from_numpy(video.get_frames_at(indices)),
    )
    sampled, timestamps = mimo_v2._decode_frames_and_timestamps(video, {"nframes": 4})
    assert sampled.shape == (4, 3, 64, 64)
    torch.testing.assert_close(timestamps, torch.tensor([0.0, 0.5, 1.5, 2.5]))
    video.close()
