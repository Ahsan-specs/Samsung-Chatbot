import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import tempfile
import json
import shutil
from dotenv import load_dotenv

load_dotenv()

from src.document_processor import DocumentProcessor
from src.retriever import HybridRetriever
from src.multimodal import MultiModalProcessor
from src.agent import SupportAgent

app = FastAPI(title="Samsung AI Support API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
processor = None
retriever = None
agent = None
multimodal = None

def init_system():
    global processor, retriever, agent, multimodal
    try:
        cerebras_key = os.environ.get("CEREBRAS_API_KEY")
        groq_key = os.environ.get("GROQ_API_KEY")
        
        if not cerebras_key:
            print("WARNING: CEREBRAS_API_KEY not found in environment.")
        if not groq_key:
            print("WARNING: GROQ_API_KEY not found in environment (needed for Voice/Vision).")

        processor = DocumentProcessor(output_dir="data/processed")
        kb_loaded = processor.load_kb()

        if kb_loaded:
            retriever = HybridRetriever(processor, top_k=5, similarity_threshold=0.18)
        else:
            retriever = None

        agent = SupportAgent(api_key=cerebras_key, retriever=retriever)
        multimodal = MultiModalProcessor(api_key=groq_key)
        print("System initialized successfully")
    except Exception as e:
        print(f"Error initializing system: {e}")

init_system()

# ─── Models ──────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: List[Message] = []

# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not agent:
        raise HTTPException(status_code=500, detail="System not initialized. Check API keys.")

    history = [{"role": m.role, "content": m.content} for m in request.history]

    def event_stream():
        try:
            for item in agent.generate_response(request.query, history, input_type="text", image_analysis=None):
                yield f"data: {json.dumps(item)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    """
    Groq Whisper transcription — accepts webm/ogg/wav from the browser MediaRecorder.
    """
    if not multimodal:
        raise HTTPException(status_code=500, detail="Multimodal processor not initialized.")

    filename = audio.filename or "recording.webm"
    ext = os.path.splitext(filename)[1]
    if not ext:
        # Guess from content-type
        ct = audio.content_type or ""
        if "ogg" in ct:
            ext = ".ogg"
        elif "webm" in ct:
            ext = ".webm"
        else:
            ext = ".webm"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(audio.file, tmp)
            tmp_path = tmp.name

        # Use Groq Whisper directly — it handles webm/ogg/wav
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()

        # Save a debug copy so user can play it and check if mic works
        debug_path = os.path.join(os.getcwd(), "debug_audio" + ext)
        with open(debug_path, "wb") as f_debug:
            f_debug.write(audio_bytes)
            
        print(f"[Audio Debug] Received {len(audio_bytes)} bytes. Format: {ext}. Saved copy to: {debug_path}")

        if len(audio_bytes) < 500:
            raise HTTPException(status_code=422, detail="Recording too short. Please speak for at least 1 second.")

        transcription = multimodal.client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-large-v3-turbo",
            response_format="text",
            language="en",
            prompt="This is a user asking a support query about a Samsung product."
        )
        transcript = str(transcription).strip()
        print(f"[Audio Debug] Raw Whisper Output: '{transcript}'")

        # Filter out common Whisper model hallucinations on silence/noise
        hallucinations = ["Thank you.", "Thank you", "Thanks.", "Thanks", "you", "Transcription by Amara.org"]
        if transcript in hallucinations or len(transcript) < 2:
            raise HTTPException(status_code=422, detail=f"No speech detected. (Raw: {transcript})")

        return {"transcript": transcript}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
