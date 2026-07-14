"""
Audio transcription via faster-whisper (CTranslate2 backend). Chosen over
openai-whisper because it's far lighter on RAM/CPU and has good ARM64 wheels
-- important for a Pi 5 with 4GB RAM.
"""
from faster_whisper import WhisperModel

import config


class Transcriber:
    def __init__(self, model_size=config.WHISPER_MODEL_SIZE,
                 compute_type=config.WHISPER_COMPUTE_TYPE):
        # device="cpu" explicitly -- Pi 5 has no CUDA
        self.model = WhisperModel(model_size, device="cpu", compute_type=compute_type)

    def transcribe(self, audio_path):
        """Returns list of {start, end, text} segments."""
        segments, _info = self.model.transcribe(audio_path, beam_size=1, vad_filter=True)
        results = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                results.append({"start": seg.start, "end": seg.end, "text": text})
        return results
