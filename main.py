#!/usr/bin/env python3
"""
Local video RAG pipeline -- CLI.

Usage:
    python main.py ingest path/to/video.mp4 [more videos...]
    python main.py query "what color was the car that drove by?"
    python main.py query "summarize what happened" --video my_clip
"""
import argparse

from pipeline.indexer import VideoIndexer
from pipeline.retriever import Retriever
from pipeline.generator import generate_answer


def cmd_ingest(args):
    indexer = VideoIndexer()
    for video_path in args.videos:
        print(f"\n=== Indexing {video_path} ===")
        indexer.index_video(video_path)


def cmd_query(args):
    retriever = Retriever()
    hits = retriever.query(args.question, top_k=args.top_k, video_id=args.video)

    if not hits:
        print("No relevant evidence found in the index. Have you run `ingest` yet?")
        return

    print("\n--- Retrieved evidence ---")
    for h in hits:
        print(f"[{h['meta']['timestamp_str']}] ({h['score']:.2f}) {h['text']}")

    print("\n--- Generating answer via Ollama ---")
    answer = generate_answer(args.question, hits)
    print(f"\nAnswer:\n{answer}")


def main():
    parser = argparse.ArgumentParser(description="Local, on-device video RAG for Raspberry Pi 5")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Process and index one or more videos")
    p_ingest.add_argument("videos", nargs="+", help="Path(s) to video file(s)")
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query", help="Ask a question about indexed videos")
    p_query.add_argument("question", help="Natural language question")
    p_query.add_argument("--video", default=None, help="Restrict to one video_id (filename without extension)")
    p_query.add_argument("--top-k", type=int, default=6, dest="top_k")
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
