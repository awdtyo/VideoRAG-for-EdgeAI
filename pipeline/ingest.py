"""
Reads a video file, samples frames at config.SAMPLE_FPS, and extracts the
audio track to a WAV file via ffmpeg (must be installed: `sudo pacman -S
ffmpeg` on Arch).
"""
import os
import subprocess

import cv2

import config


def sample_frames(video_path):
    """Yields (timestamp_seconds, frame_bgr) at config.SAMPLE_FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, round(native_fps / config.SAMPLE_FPS))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            h, w = frame.shape[:2]
            scale = config.MAX_FRAME_DIM / max(h, w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            timestamp = idx / native_fps
            yield timestamp, frame
        idx += 1
    cap.release()


def extract_audio(video_path, out_dir=config.AUDIO_DIR):
    """Extracts mono 16kHz WAV audio (whisper's preferred format) via ffmpeg."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(out_dir, f"{base}.wav")

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(audio_path):
        # video may have no audio track -- not fatal, just skip transcription
        return None
    return audio_path
