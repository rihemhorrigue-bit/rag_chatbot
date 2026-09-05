import sys
import argparse
from pathlib import Path

# Import logic from your previous modules
try:
    from pdf_chunker import process_pdf, link_global_neighbors, DEFAULT_PDFS
    from embedder_saver import EmbedderSaver
except ImportError as e:
    print(f"Error: Could not import project modules. Ensure 'pdf_chunker.py' and 'embedder_saver.py' are in this folder.")
    print(f"Details: {e}")
    sys.exit(1)

def run_pipeline(input_pdfs, max_tokens, overlap_tokens, db_path, collection_name):
    """
    Executes the full pipeline:
    1. Parsing/Chunking
    2. Linking Neighbors
    3. Embedding & Saving to ChromaDB
    """
    
    all_chunks = []
    
    # --- STEP 1: CHUNKING ---
    print("--- Phase 1: Chunking ---")
    for pdf_path in input_pdfs:
        p = Path(pdf_path)
        if not p.exists():
            print(f"Skipping: {p} (File not found)")
            continue
            
        print(f"Processing: {p.name}...")
        # We call the process_pdf function from your chunker script
        chunks, stats = process_pdf(
            path=p,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            min_tokens=80,
            table_mode="auto",
            emit_markdown_dir=None # Change to Path("./data/md") if you want debug files
        )
        all_chunks.extend(chunks)
        print(f"  -> Generated {len(chunks)} chunks.")

    if not all_chunks:
        print("No chunks generated. Check your input PDF paths.")
        return

    # Link the chunks so they know who their 'next' and 'previous' neighbors are
    link_global_neighbors(all_chunks)

    # --- STEP 2: EMBEDDING & SAVING ---
    print("\n--- Phase 2: Embedding & Saving to ChromaDB ---")
    # Initialize the engine from our second script
    db_engine = EmbedderSaver(db_path=db_path, collection_name=collection_name)
    
    # Save the chunks (this generates embeddings automatically)
    db_engine.save_chunks_to_db(all_chunks)
    
    print("\n" + "="*30)
    print("PIPELINE COMPLETE")
    print(f"Total chunks stored: {len(all_chunks)}")
    print(f"Database location: {db_path}")
    print("="*30)

def main():
    parser = argparse.ArgumentParser(description="End-to-End RAG Ingestion Pipeline")
    parser.add_argument("--inputs", action="append", help="Path to PDF. Repeat for multiple.")
    parser.add_argument("--db-path", default="./chroma_db", help="Folder for ChromaDB")
    parser.add_argument("--collection", default="small_business_kb", help="Chroma collection name")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    
    args = parser.parse_args()
    
    # Use provided inputs or fall back to your DEFAULT_PDFS list
    input_paths = args.inputs if args.inputs else DEFAULT_PDFS
    
    run_pipeline(
        input_pdfs=input_paths,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap,
        db_path=args.db_path,
        collection_name=args.collection
    )

if __name__ == "__main__":
    main()