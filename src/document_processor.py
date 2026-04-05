"""
Document Processor Module — Samsung Agentic RAG
Handles ingestion of JSON product data and PDF documents.
Builds FAISS vector index + NetworkX knowledge graph (GraphRAG).
"""

import os
import json
import fitz  # PyMuPDF
import networkx as nx
import faiss
import numpy as np
import spacy
import pickle
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional


class DocumentProcessor:
    """
    Ingests Samsung product data (JSON specs + PDFs), performs semantic chunking,
    generates embeddings, and builds a hybrid retrieval index (FAISS + GraphRAG).
    """

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("data/raw_pdfs", exist_ok=True)

        # ----- Embedding Model -----
        print("Loading Sentence Transformer model...")
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"[WARN] Failed to load sentence-transformers: {e}")
            self.embedder = None

        # ----- NLP for Entity Extraction -----
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model en_core_web_sm...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.nlp = spacy.blank("en")

        # ----- Text Splitter for PDFs -----
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # ----- State -----
        self.chunks: List[Dict] = []
        self.faiss_index: Optional[faiss.Index] = None
        self.graph: nx.Graph = nx.Graph()
        self.product_catalog: Dict[str, Dict] = {}  # Quick lookup by title

    # =====================================================================
    #  JSON PRODUCT DATA INGESTION
    # =====================================================================

    def ingest_json_folder(self, data_dir: str) -> int:
        """
        Recursively scans a directory for JSON product files and ingests them.
        Returns the number of products processed.
        """
        count = 0
        for root, dirs, files in os.walk(data_dir):
            # Skip processed and raw_pdfs directories
            if "processed" in root or "raw_pdfs" in root:
                continue
            for file in files:
                if file.endswith(".json"):
                    path = os.path.join(root, file)
                    try:
                        self._ingest_single_json(path)
                        count += 1
                    except Exception as e:
                        print(f"[WARN] Failed to process {file}: {e}")

        if count > 0:
            print(f"Ingested {count} JSON product files.")
            self._build_vector_store()
            self._build_graph()
            self.save_kb()
        return count

    def _ingest_single_json(self, file_path: str):
        """Parses a single Samsung product JSON file into semantic chunks."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        title = data.get("title", os.path.basename(file_path).replace(".json", ""))
        category = data.get("category", "Unknown")
        url = data.get("url", "")
        features = data.get("features", [])
        specs = data.get("specifications", {})

        # Store in catalog for quick lookup
        self.product_catalog[title.lower()] = data

        start_id = len(self.chunks)

        # -- Chunk 1: Product Overview --
        overview_parts = [f"Product: {title}", f"Category: {category}"]
        if url:
            overview_parts.append(f"URL: {url}")
        if features:
            overview_parts.append("Key Features:")
            for feat in features:
                if feat and len(feat.strip()) > 0:
                    overview_parts.append(f"  - {feat.strip()}")
        overview_text = "\n".join(overview_parts)

        self.chunks.append({
            "id": start_id,
            "text": overview_text,
            "metadata": {
                "source": title,
                "category": category,
                "chunk_type": "overview",
                "url": url
            }
        })

        # -- Chunk 2: Technical Specifications --
        if specs:
            spec_lines = [f"Technical Specifications for {title} ({category}):"]
            for key, value in specs.items():
                spec_lines.append(f"  {key}: {value}")
            spec_text = "\n".join(spec_lines)

            # If specs are very long, split further
            if len(spec_text) > 1200:
                spec_groups = self._split_specs_into_groups(specs, title, category)
                for i, group_text in enumerate(spec_groups):
                    self.chunks.append({
                        "id": start_id + 1 + i,
                        "text": group_text,
                        "metadata": {
                            "source": title,
                            "category": category,
                            "chunk_type": f"specs_part_{i+1}",
                            "url": url
                        }
                    })
            else:
                self.chunks.append({
                    "id": start_id + 1,
                    "text": spec_text,
                    "metadata": {
                        "source": title,
                        "category": category,
                        "chunk_type": "specifications",
                        "url": url
                    }
                })

    def _split_specs_into_groups(self, specs: dict, title: str, category: str) -> List[str]:
        """Splits large specification dictionaries into logical groups."""
        # Group specs by common prefixes/themes
        display_keys = []
        camera_keys = []
        connectivity_keys = []
        battery_keys = []
        general_keys = []
        audio_keys = []
        cooling_keys = []

        for key in specs:
            kl = key.lower()
            if any(w in kl for w in ["display", "screen", "resolution", "color depth", "technology", "size (main", "panel", "refresh", "hdr"]):
                display_keys.append(key)
            elif any(w in kl for w in ["camera", "zoom", "ois", "flash", "video recording", "slow motion"]):
                camera_keys.append(key)
            elif any(w in kl for w in ["wifi", "wi-fi", "bluetooth", "usb", "nfc", "sim", "lte", "5g", "gsm", "umts", "infra", "gps", "location", "earjack", "mhl"]):
                connectivity_keys.append(key)
            elif any(w in kl for w in ["battery", "charging", "power", "watt"]):
                battery_keys.append(key)
            elif any(w in kl for w in ["audio", "sound", "speaker", "dolby", "stereo"]):
                audio_keys.append(key)
            elif any(w in kl for w in ["cooling", "compressor", "refrigerant", "btu", "capacity", "noise"]):
                cooling_keys.append(key)
            else:
                general_keys.append(key)

        groups = []
        group_map = {
            "Display": display_keys,
            "Camera": camera_keys,
            "Connectivity": connectivity_keys,
            "Battery & Power": battery_keys,
            "Audio": audio_keys,
            "Cooling & Performance": cooling_keys,
            "General": general_keys
        }

        for group_name, keys in group_map.items():
            if keys:
                lines = [f"{group_name} Specifications for {title} ({category}):"]
                for k in keys:
                    lines.append(f"  {k}: {specs[k]}")
                groups.append("\n".join(lines))

        return groups if groups else [f"Specifications for {title}: " + str(specs)]

    # =====================================================================
    #  PDF INGESTION (legacy support)
    # =====================================================================

    def ingest_pdfs(self, pdf_dir: str):
        """Processes all PDFs in a directory."""
        all_text_chunks = []
        for file in os.listdir(pdf_dir):
            if file.endswith(".pdf"):
                path = os.path.join(pdf_dir, file)
                print(f"Processing PDF: {file}")
                text = self._extract_text_mupdf(path)
                file_chunks = self.text_splitter.create_documents(
                    [text], metadatas=[{"source": file}]
                )
                all_text_chunks.extend(file_chunks)

        if not all_text_chunks:
            print("No valid PDF data found.")
            return

        start_id = len(self.chunks)
        for i, c in enumerate(all_text_chunks):
            self.chunks.append({
                "id": start_id + i,
                "text": c.page_content,
                "metadata": {**c.metadata, "chunk_type": "pdf", "category": "Document"},
            })

        self._build_vector_store()
        self._build_graph()
        self.save_kb()

    def ingest_single_pdf(self, file_path: str):
        """Processes a single uploaded PDF."""
        text = self._extract_text_mupdf(file_path)
        file_name = os.path.basename(file_path)

        docs = self.text_splitter.create_documents(
            [text], metadatas=[{"source": file_name}]
        )
        start_id = len(self.chunks)
        for i, d in enumerate(docs):
            self.chunks.append({
                "id": start_id + i,
                "text": d.page_content,
                "metadata": {
                    **d.metadata,
                    "chunk_type": "pdf",
                    "category": "Uploaded Document"
                },
            })

        self._build_vector_store()
        self._build_graph()
        self.save_kb()

    def _extract_text_mupdf(self, file_path: str) -> str:
        """Extracts text from PDF preserving layout."""
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return text

    # =====================================================================
    #  VECTOR STORE (FAISS)
    # =====================================================================

    def _build_vector_store(self):
        """Creates FAISS index from all chunks."""
        if not self.embedder or not self.chunks:
            return

        texts = [c["text"] for c in self.chunks]
        print(f"Encoding {len(texts)} chunks...")
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        dimension = embeddings.shape[1]
        # Using IndexFlatIP (Inner Product) after normalizing for cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)
        print(f"FAISS index built with {self.faiss_index.ntotal} vectors (dim={dimension}).")

    # =====================================================================
    #  KNOWLEDGE GRAPH (GraphRAG)
    # =====================================================================

    def _build_graph(self):
        """
        Builds a knowledge graph linking:
        - Products → Categories
        - Products → Chunks
        - Products → Entities (specs, features)
        - Entities → Chunks (co-occurrence)
        """
        self.graph = nx.Graph()

        for chunk in self.chunks:
            chunk_id = chunk["id"]
            text = chunk["text"]
            source = chunk["metadata"].get("source", "unknown")
            category = chunk["metadata"].get("category", "Unknown")
            chunk_type = chunk["metadata"].get("chunk_type", "unknown")

            chunk_node = f"CHUNK:{chunk_id}"
            prod_node = f"PRODUCT:{source}"
            cat_node = f"CATEGORY:{category}"

            # Add chunk node
            self.graph.add_node(chunk_node, type="chunk", text=text[:500],
                                source=source, category=category)

            # Add product node and link to chunk
            self.graph.add_node(prod_node, type="product", name=source, category=category)
            self.graph.add_edge(chunk_node, prod_node, relation="belongs_to")

            # Add category node and link to product
            self.graph.add_node(cat_node, type="category")
            self.graph.add_edge(prod_node, cat_node, relation="in_category")

            # Extract entities using spaCy
            doc = self.nlp(text[:2000])  # Limit to avoid heavy processing
            entities = set()
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "QUANTITY",
                                   "CARDINAL", "GPE", "FAC"]:
                    clean = ent.text.lower().strip()
                    if len(clean) >= 3:
                        entities.add(clean)

            # Extract Samsung-specific patterns
            samsung_patterns = re.findall(
                r'\b(galaxy\s+\w+(?:\s+\w+)?|samsung\s+\w+|sm-\w+|qa\w+|ua\w+|'
                r'qled|oled|neo qled|crystal uhd|4k|8k|5g|windfree|'
                r'smartthings|tizen|one ui|bixby)\b',
                text.lower()
            )
            entities.update(samsung_patterns)

            # Extract noun chunks for product features
            for nc in doc.noun_chunks:
                nc_text = nc.text.lower().strip()
                if 3 <= len(nc_text.split()) <= 5 and len(nc_text) >= 5:
                    entities.add(nc_text)

            # Link entities to chunks
            for ent in entities:
                ent_node = f"ENT:{ent}"
                if not self.graph.has_node(ent_node):
                    self.graph.add_node(ent_node, type="entity")
                self.graph.add_edge(ent_node, chunk_node, relation="mentioned_in")

        print(f"Knowledge Graph: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges.")

    # =====================================================================
    #  PERSISTENCE
    # =====================================================================

    def save_kb(self):
        """Saves chunks, FAISS index, graph, and catalog to disk."""
        if self.faiss_index:
            faiss.write_index(self.faiss_index,
                              os.path.join(self.output_dir, "vector_index.faiss"))

        with open(os.path.join(self.output_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

        with open(os.path.join(self.output_dir, "graph.pkl"), "wb") as f:
            pickle.dump(self.graph, f)

        with open(os.path.join(self.output_dir, "catalog.pkl"), "wb") as f:
            pickle.dump(self.product_catalog, f)

        print(f"Knowledge Base saved ({len(self.chunks)} chunks).")

    def load_kb(self) -> bool:
        """Loads KB from disk. Returns True if successful."""
        try:
            faiss_path = os.path.join(self.output_dir, "vector_index.faiss")
            chunks_path = os.path.join(self.output_dir, "chunks.pkl")
            graph_path = os.path.join(self.output_dir, "graph.pkl")
            catalog_path = os.path.join(self.output_dir, "catalog.pkl")

            if not os.path.exists(chunks_path):
                return False

            self.faiss_index = faiss.read_index(faiss_path)
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            with open(graph_path, "rb") as f:
                self.graph = pickle.load(f)
            if os.path.exists(catalog_path):
                with open(catalog_path, "rb") as f:
                    self.product_catalog = pickle.load(f)

            print(f"Knowledge Base loaded: {len(self.chunks)} chunks, "
                  f"{self.graph.number_of_nodes()} graph nodes.")
            return True
        except Exception as e:
            print(f"[WARN] Could not load KB: {e}")
            return False

    def get_stats(self) -> Dict:
        """Returns KB statistics for the UI."""
        categories = set()
        products = set()
        for ch in self.chunks:
            categories.add(ch["metadata"].get("category", "Unknown"))
            products.add(ch["metadata"].get("source", "Unknown"))

        return {
            "total_chunks": len(self.chunks),
            "total_products": len(products),
            "total_categories": len(categories),
            "graph_nodes": self.graph.number_of_nodes() if self.graph else 0,
            "graph_edges": self.graph.number_of_edges() if self.graph else 0,
            "faiss_vectors": self.faiss_index.ntotal if self.faiss_index else 0,
            "categories": sorted(list(categories)),
        }


if __name__ == "__main__":
    processor = DocumentProcessor()
    count = processor.ingest_json_folder("data")
    print(f"Processed {count} files.")
    stats = processor.get_stats()
    print(f"Stats: {stats}")
