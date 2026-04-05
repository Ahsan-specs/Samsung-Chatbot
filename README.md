# Samsung Galaxy AI Support Assistant 🌌

A premium, high-end AI Support interface designed strictly according to the **Samsung One UI** design language. This project leverages an advanced **Agentic RAG** (Retrieval-Augmented Generation) pipeline to provide accurate, grounded support for Samsung products.

## 🚀 Key Features

- **Premium Samsung Branding**: Full-width responsive landing page with a "Galaxy Dark" theme and a clean "White & Blue" chatbot widget.
- **Agentic RAG Architecture**:
  - **Hybrid Retrieval**: Combines FAISS vector search with GraphRAG (NetworkX) for deep context understanding.
  - **Smart Web Fallback**: Proactively asks for permission to search the web (via DuckDuckGo) when local manuals are insufficient, restricted strictly to `samsung.com`.
- **High Performance**: Powered by the **Cerebras Llama-3.1-8b** model for sub-second inference speeds.
- **Multi-modal Support**: Integrated Voice-to-Text (STT) and Text-to-Speech (TTS) for a hands-free experience.
- **Grounding & Guardrails**: Strict rules to prevent hallucinations, pricing inaccuracies, and out-of-scope queries (Samsung products only).

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python 3.11)
- **Frontend**: React + Vite + Vanilla CSS
- **Vector DB**: FAISS
- **Graph DB**: NetworkX (Local GraphRAG)
- **AI Models**: Llama 3.1 8B (via Cerebras), STT/Image Analysis (via Groq/OpenAI compatible)
- **Styling**: Google Fonts (Outfit, Inter), Lucide Icons

## 💻 Local Setup & Running

### Prerequisites
- Python 3.11+
- Node.js & npm
- [Cerebras API Key](https://cloud.cerebras.ai/)
- [Groq API Key](https://console.groq.com/) (for STT/Multimodal)

### 1. Environment Configuration
Create a `.env` file in the `Data/` directory:
```env
CEREBRAS_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

### 2. Backend Installation (macOS/Linux)
```bash
cd Data
python3.11 -m venv venv_mac
source venv_mac/bin/activate
pip install -r requirements.txt
python server.py
```

### 3. Backend Installation (Windows)
```powershell
cd Data
python -m venv venv_win
.\venv_win\Scripts\activate
pip install -r requirements.txt
python server.py
```

### 4. Frontend Installation
```bash
cd Data/frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

## 📂 Project Structure
- `/src`: Core agentic logic, retrievers, and document processors.
- `/Data`: Project root containing backend server and knowledge base.
- `/frontend`: React source code and premium styling.
- `storage.sqlite`: Local vector and graph storage.

---
*Created with ❤️ for the Samsung Galaxy Ecosystem.*
