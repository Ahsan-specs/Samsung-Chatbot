"""
Hybrid Retriever Module — Samsung Agentic RAG
Combines FAISS vector search with GraphRAG context expansion.
Includes similarity thresholding and source citation.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
from .document_processor import DocumentProcessor


class HybridRetriever:
    """
    Retrieves relevant context using a hybrid approach:
    1. FAISS cosine-similarity vector search (top-k)
    2. GraphRAG expansion via entity/product neighbors
    3. Similarity thresholding for out-of-scope detection
    """

    def __init__(self, processor: DocumentProcessor, top_k: int = 5,
                 similarity_threshold: float = 0.25):
        self.processor = processor
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold  # Min cosine similarity

    def retrieve(self, query: str) -> Dict:
        """
        Retrieves and formats context using Vector + GraphRAG hybrid approach.
        
        Returns dict with:
        - context: formatted string for LLM
        - sources: list of source documents used
        - scores: similarity scores
        - is_relevant: whether the query matched KB content
        - chunks_used: number of chunks retrieved
        """
        result = {
            "context": "",
            "sources": [],
            "scores": [],
            "is_relevant": False,
            "chunks_used": 0,
            "graph_expanded": 0,
        }

        if not self.processor.faiss_index or not self.processor.chunks:
            result["context"] = "Knowledge base not initialized or empty."
            return result

        if not self.processor.embedder:
            result["context"] = "Embedding model not available."
            return result

        # ---- Step 1: Vector Search ----
        query_embedding = self.processor.embedder.encode(
            [query], convert_to_numpy=True
        )
        # Normalize for cosine similarity (IndexFlatIP)
        import faiss
        faiss.normalize_L2(query_embedding)

        scores, indices = self.processor.faiss_index.search(
            query_embedding, min(self.top_k, self.processor.faiss_index.ntotal)
        )

        # Collect chunks that pass threshold
        retrieved_chunks = []
        vector_chunk_ids = set()

        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            similarity = float(scores[0][i])
            if similarity >= self.similarity_threshold:
                chunk = self.processor.chunks[idx]
                retrieved_chunks.append({
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "source": chunk["metadata"].get("source", "Unknown"),
                    "category": chunk["metadata"].get("category", "Unknown"),
                    "chunk_type": chunk["metadata"].get("chunk_type", "unknown"),
                    "url": chunk["metadata"].get("url", ""),
                    "similarity": similarity,
                })
                vector_chunk_ids.add(chunk["id"])

        if not retrieved_chunks:
            # Everything below threshold — out of scope
            return result

        result["is_relevant"] = True
        result["scores"] = [c["similarity"] for c in retrieved_chunks]

        # ---- Step 2: GraphRAG Expansion ----
        graph = self.processor.graph
        expanded_chunks = []
        seen_ids = set(vector_chunk_ids)

        for ch_id in list(vector_chunk_ids):
            node = f"CHUNK:{ch_id}"
            if not graph.has_node(node):
                continue

            # Get direct neighbors (entities, product nodes)
            for neighbor in graph.neighbors(node):
                if neighbor.startswith("ENT:") or neighbor.startswith("PRODUCT:"):
                    # Get chunks related to same entity/product
                    for second_hop in graph.neighbors(neighbor):
                        if second_hop.startswith("CHUNK:"):
                            other_id = int(second_hop.split(":")[1])
                            if other_id not in seen_ids:
                                seen_ids.add(other_id)
                                other_chunk = self.processor.chunks[other_id]
                                expanded_chunks.append({
                                    "id": other_id,
                                    "text": other_chunk["text"],
                                    "source": other_chunk["metadata"].get("source", "Unknown"),
                                    "category": other_chunk["metadata"].get("category", "Unknown"),
                                    "chunk_type": other_chunk["metadata"].get("chunk_type", "unknown"),
                                    "url": other_chunk["metadata"].get("url", ""),
                                    "similarity": 0.0,  # Graph-expanded, no direct score
                                    "expansion_source": neighbor,
                                })

        # Limit graph expansion to top 3 to avoid token overflow
        expanded_chunks = expanded_chunks[:3]
        result["graph_expanded"] = len(expanded_chunks)

        # ---- Step 3: Format Context ----
        sources_set = set()
        context_parts = []

        for i, chunk in enumerate(retrieved_chunks, 1):
            source_label = f"{chunk['source']} ({chunk['category']})"
            sources_set.add(source_label)
            context_parts.append(chunk['text'])

        if expanded_chunks:
            for i, chunk in enumerate(expanded_chunks, 1):
                source_label = f"{chunk['source']} ({chunk['category']})"
                sources_set.add(source_label)
                context_parts.append(chunk['text'])

        result["context"] = "\n".join(context_parts)
        result["sources"] = sorted(list(sources_set))
        result["chunks_used"] = len(retrieved_chunks) + len(expanded_chunks)

        return result

    def search_by_category(self, category: str, query: str) -> Dict:
        """
        Filtered search: only considers chunks from a specific category.
        Useful for the agent to narrow down search scope.
        """
        if not self.processor.faiss_index or not self.processor.chunks:
            return {"context": "", "sources": [], "is_relevant": False}

        # Get all chunks matching the category (to verify the category exists)
        category_exists = any(c["metadata"].get("category", "").lower() == category.lower() for c in self.processor.chunks)
        if not category_exists:
            return {"context": f"No products found in category: {category}",
                    "sources": [], "is_relevant": False}

        # Embed query ONCE
        query_embedding = self.processor.embedder.encode(
            [query], convert_to_numpy=True
        )
        import faiss as faiss_lib
        faiss_lib.normalize_L2(query_embedding)

        # Search FAISS for top 100 results (extremely fast)
        search_depth = min(100, self.processor.faiss_index.ntotal)
        scores, indices = self.processor.faiss_index.search(
            query_embedding, search_depth
        )

        sources_set = set()
        context_parts = [f"═══ CATEGORY SEARCH: {category} ═══"]
        chunks_used = 0

        # Filter the fast FAISS results by category
        for i, idx in enumerate(indices[0]):
            if idx == -1 or chunks_used >= self.top_k:
                continue
            
            similarity = float(scores[0][i])
            if similarity < self.similarity_threshold:
                continue
                
            chunk = self.processor.chunks[idx]
            if chunk["metadata"].get("category", "").lower() == category.lower():
                source_label = f"{chunk['metadata'].get('source', 'Unknown')} ({category})"
                sources_set.add(source_label)
                context_parts.append(chunk['text'])
                chunks_used += 1

        return {
            "context": "\n".join(context_parts),
            "sources": sorted(list(sources_set)),
            "is_relevant": len(sources_set) > 0,
            "chunks_used": len(sources_set),
        }
