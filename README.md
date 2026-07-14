# Local Video RAG Pipeline for Raspberry Pi 5

Fully local, multimodal video question-answering. Nothing leaves the device —
no cloud APIs, no `ultralytics`. Sized to run on a Pi 5 with 4GB RAM.

## Architecture

```
video.mp4
   ├── frames (sampled @ 1fps) ──► YOLOX-Nano (ONNX) ──► object labels
   │                           └─► Tesseract OCR (every Nth frame) ──► on-screen text
   │                           └─► MobileNetV3 (ONNX, optional) ──► visual embedding
   └── audio track ──► faster-whisper (tiny/base) ──► timestamped transcript

All of the above → short timestamped text documents
                 → embedded with BGE-small-en-v1.5
                 → stored in ChromaDB (local, on-disk)

query (text) → embed → retrieve top-k timestamped docs → Ollama (qwen2.5:3b)
            → grounded, timestamp-cited answer
```

Why these components:
- **YOLOX-Nano over ultralytics/YOLO**: Apache-2.0, small (~7MB), ONNX-native,
  no `ultralytics` package dependency, runs comfortably on 4 CPU threads.
- **faster-whisper over openai-whisper**: CTranslate2 backend, int8
  quantization, far lower RAM/CPU on ARM.
- **BGE-small / MiniLM-class text embedder**: ~130MB, good enough quality for
  RAG over short captions/transcripts, fast on CPU.
- **ChromaDB**: embedded, no separate server process, persists to disk.
- **Ollama for generation**: keeps the LLM step decoupled — you can point it
  at a bigger model on another machine on your LAN if the Pi is loaded
  elsewhere in the pipeline (e.g. running your existing image RAG pipeline
  concurrently).

## Setup

```bash
cd video_rag_pi
chmod +x setup.sh download_models.sh
./setup.sh
```

This installs system deps (ffmpeg, tesseract), creates a venv, installs
Python deps, and downloads the two ONNX models. If either model URL has
moved, `download_models.sh` prints the manual fallback locations.

Pull a small local LLM for generation:
```bash
ollama pull qwen2.5:3b
```

## Usage

```bash
source .venv/bin/activate

# Index one or more videos
python main.py ingest data/videos/clip1.mp4 data/videos/clip2.mp4

# Ask questions
python main.py query "how many people appear in the video?"
python main.py query "what does the sign in the shop say?" --video clip1
```

Each answer is generated from retrieved, timestamped evidence and the model
is prompted to cite timestamps — check `--top-k` retrieved lines printed
above the answer to see exactly what it was grounded on.

## Tuning for 4GB RAM

Everything below lives in `config.py`:

| Setting | Effect |
|---|---|
| `SAMPLE_FPS` | Lower = fewer frames to process = less RAM/CPU. Start at 0.5 for long videos. |
| `ENABLE_VISUAL_EMBEDDINGS` | Set `False` to skip MobileNetV3 entirely if you only need text-grounded retrieval. |
| `ENABLE_OCR` / `OCR_EVERY_N_FRAMES` | Tesseract is CPU-heavy; raise the N or disable if you don't need on-screen text. |
| `WHISPER_MODEL_SIZE` | `tiny` uses far less RAM than `base`; drop to `tiny` first if you hit OOM. |
| `MAX_FRAME_DIM` | Frames are downscaled before detection — lower this to cut detector cost. |

The pipeline processes one video at a time, sequentially, and loads all
models once per `ingest`/`query` run rather than per-frame — this keeps peak
RAM predictable. If you still hit memory pressure, run `ingest` and `query`
as two separate invocations (already the default) rather than combining
them, so model sets don't overlap.

## Extending

- Swap `qwen2.5:3b` for `phi3:mini` or `llama3.2:3b` in `config.py` —
  anything Ollama can serve works.
- The visual embeddings (if enabled) are stored per-frame in metadata; you
  can add an image-to-image query mode by embedding a query frame with
  `MobileNetEmbedder` and doing a Chroma vector search against those, same
  pattern as the text retriever.
- This reuses your existing model choices where useful (Whisper, BGE
  embeddings, ChromaDB) so it should feel consistent with your video RAG
  work under Prof. Nandi — the five-branch (Temporal/Spatial/Spatio-Temporal/
  Counting/Audio) split from that project could be layered on top of this by
  routing queries to different retrieval strategies before this generation
  step, if you want to extend it that way.
