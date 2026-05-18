"""
RAG Pipeline — CLI
Usage:
  python main.py ingest  --path data/
  python main.py query   --question "What is RAG?"
  python main.py chat                          # interactive REPL
"""

import argparse
import sys
from src.rag_pipeline import RAGPipeline

INDEX_DIR = "index"


def ingest(args):
    rag = RAGPipeline()
    rag.ingest(args.path)
    rag.save_index(INDEX_DIR)
    print("✅ Ingestion complete.")


def query(args):
    rag = RAGPipeline()
    rag.load_index(INDEX_DIR)
    result = rag.query(args.question, verbose=args.verbose)
    print(f"\n💬 Answer:\n{result['answer']}")
    print(f"\n📎 Sources: {', '.join(result['sources'])}")


def chat(args):
    rag = RAGPipeline()
    rag.load_index(INDEX_DIR)
    print("\n🤖 RAG Chat — type 'exit' to quit\n" + "─" * 40)
    while True:
        try:
            q = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!"); sys.exit(0)
        if q.lower() in {"exit", "quit", "q"}:
            print("Bye!"); break
        if not q:
            continue
        result = rag.query(q)
        print(f"\nAssistant: {result['answer']}")
        print(f"Sources  : {', '.join(result['sources'])}")


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="Ingest documents into the index")
    p_ingest.add_argument("--path", required=True, help="File or directory to ingest")

    p_query = sub.add_parser("query", help="One-shot query")
    p_query.add_argument("--question", required=True)
    p_query.add_argument("--verbose", action="store_true")

    sub.add_parser("chat", help="Interactive chat loop")

    args = parser.parse_args()
    if args.cmd == "ingest":   ingest(args)
    elif args.cmd == "query":  query(args)
    elif args.cmd == "chat":   chat(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
