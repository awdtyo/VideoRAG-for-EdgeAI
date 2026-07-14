"""
Orchestrates a single video through: frame sampling -> detection -> (optional)
OCR -> (optional) visual embedding, plus audio -> transcription. All
resulting evidence is turned into short text "documents" tagged with a
timestamp and embedded into ChromaDB for later retrieval.

Kept single-pass and sequential (no batching) to stay within the Pi 5's
4GB RAM budget.
"""
import os
import uuid

import chromadb
from sentence_transformers import SentenceTransformer

import config
from models.detector import YoloxNanoDetector
from models.embedder import MobileNetEmbedder
from models.ocr_engine import extract_text
from models.transcriber import Transcriber
from pipeline.ingest import sample_frames, extract_audio


def _fmt_ts(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class VideoIndexer:
    def __init__(self):
        print("Loading models (this happens once per run)...")
        self.detector = YoloxNanoDetector()
        self.embedder = MobileNetEmbedder() if config.ENABLE_VISUAL_EMBEDDINGS else None
        self.transcriber = Transcriber()
        self.text_embedder = SentenceTransformer(config.TEXT_EMBED_MODEL, device="cpu")

        client = chromadb.PersistentClient(path=config.DB_DIR)
        self.collection = client.get_or_create_collection(config.CHROMA_COLLECTION)

    def index_video(self, video_path):
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        docs, metadatas, ids = [], [], []

        # ---- visual stream: detection (+ optional OCR / visual embedding) ----
        for i, (ts, frame) in enumerate(sample_frames(video_path)):
            detections = self.detector.detect(frame)
            labels = sorted({d["label"] for d in detections})

            ocr_text = ""
            if config.ENABLE_OCR and i % config.OCR_EVERY_N_FRAMES == 0:
                ocr_text = extract_text(frame)

            if not labels and not ocr_text:
                continue  # skip empty frames, nothing to index

            parts = []
            if labels:
                parts.append(f"Visible objects: {', '.join(labels)}.")
            if ocr_text:
                parts.append(f"On-screen text: {ocr_text}")
            doc_text = " ".join(parts)

            visual_vec = self.embedder.embed(frame) if self.embedder else None

            docs.append(doc_text)
            metadatas.append({
                "video_id": video_id,
                "type": "frame",
                "timestamp": ts,
                "timestamp_str": _fmt_ts(ts),
                "has_visual_embedding": visual_vec is not None,
            })
            ids.append(str(uuid.uuid4()))
            print(f"  [{video_id}] frame @ {_fmt_ts(ts)}: {doc_text[:80]}")

        # ---- audio stream: transcription ----
        audio_path = extract_audio(video_path)
        if audio_path:
            print(f"  [{video_id}] transcribing audio...")
            for seg in self.transcriber.transcribe(audio_path):
                docs.append(f"Spoken: {seg['text']}")
                metadatas.append({
                    "video_id": video_id,
                    "type": "speech",
                    "timestamp": seg["start"],
                    "timestamp_str": _fmt_ts(seg["start"]),
                })
                ids.append(str(uuid.uuid4()))

        if not docs:
            print(f"  [{video_id}] nothing indexable found.")
            return 0

        # ---- embed + store ----
        embeddings = self.text_embedder.encode(docs, normalize_embeddings=True).tolist()
        self.collection.add(
            documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids
        )
        print(f"  [{video_id}] indexed {len(docs)} documents.")
        return len(docs)
