"""Build Knowledge Base from data/ folder."""
from src.document_processor import DocumentProcessor
import time

start = time.time()
p = DocumentProcessor(output_dir="data/processed")
count = p.ingest_json_folder("data")
p.ingest_pdfs("data/raw_pdfs")
elapsed = time.time() - start

stats = p.get_stats()
print("\n=== KB BUILD COMPLETE ===")
print(f"Products: {stats['total_products']}")
print(f"Categories: {stats['total_categories']}")
print(f"Chunks: {stats['total_chunks']}")
print(f"FAISS Vectors: {stats['faiss_vectors']}")
print(f"Graph Nodes: {stats['graph_nodes']}")
print(f"Graph Edges: {stats['graph_edges']}")
print(f"Time: {elapsed:.1f}s")
print("Categories:", stats['categories'])
