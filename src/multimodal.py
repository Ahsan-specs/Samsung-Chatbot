"""
Multi-Modal Processor — Samsung Agentic RAG
Handles image analysis (OCR/Vision) and voice transcription.
Uses Groq's LLaMA Vision and Whisper APIs.
"""

import os
import io
import wave
import struct
import base64
from groq import Groq
from typing import Optional


class MultiModalProcessor:
    """
    Processes non-text inputs:
    - Images → OCR/Vision description via Groq LLaMA Vision
    - Audio  → Transcription via Groq Whisper
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required for MultiModalProcessor (Vision/Whisper).")
        self.client = Groq(api_key=self.api_key)

    def _validate_audio(self, audio_path: str) -> tuple:
        """
        Validates that the audio file has actual audio content.
        Returns (is_valid, message).
        """
        try:
            file_size = os.path.getsize(audio_path)
            # WAV header is 44 bytes; need at least some audio data
            if file_size < 1000:
                return False, "Recording too short. Please hold the mic button and speak clearly."

            # Try to read WAV properties to check duration
            try:
                with wave.open(audio_path, 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate) if rate > 0 else 0
                    if duration < 0.5:
                        return False, f"Recording too short ({duration:.1f}s). Please speak for at least 1 second."
            except wave.Error:
                # Not a standard WAV — might still work with Whisper
                pass

            return True, "OK"
        except Exception as e:
            return False, f"Audio validation error: {e}"

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribes an audio file using Groq's Whisper model.
        Returns transcribed text or error message.
        """
        # Validate audio first
        is_valid, validation_msg = self._validate_audio(audio_path)
        if not is_valid:
            return f"Audio transcription failed: {validation_msg}"

        try:
            with open(audio_path, "rb") as file:
                audio_data = file.read()

            # Try with whisper-large-v3-turbo first (faster, more reliable)
            try:
                transcription = self.client.audio.transcriptions.create(
                    file=("recording.wav", audio_data),
                    model="whisper-large-v3-turbo",
                    response_format="text",
                    language="en"
                )
                result = str(transcription).strip()
                if result and len(result) > 1:
                    return result
            except Exception:
                pass

            # Fallback to whisper-large-v3
            transcription = self.client.audio.transcriptions.create(
                file=("recording.wav", audio_data),
                model="whisper-large-v3",
                response_format="text",
                language="en"
            )
            result = str(transcription).strip()
            if result and len(result) > 1:
                return result
            else:
                return "Audio transcription failed: No speech detected. Please speak clearly and try again."
        except Exception as e:
            return f"Audio transcription failed: {e}"

    def process_image(self, image_path: str,
                      prompt: Optional[str] = None) -> str:
        """
        Analyzes an image using Groq's LLaMA Vision model.
        Extracts text, identifies Samsung products, and describes visual content.
        Returns the analysis text or error message.
        """
        if prompt is None:
            prompt = (
                "Analyze this image in the context of Samsung product support. "
                "1. Extract ALL visible text (error messages, model numbers, settings). "
                "2. Identify the Samsung product/device if visible. "
                "3. Describe any visual symptoms, errors, or issues shown. "
                "4. Note any relevant indicators (LED colors, screen states, damage). "
                "Be specific and detailed."
            )

        try:
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Determine MIME type
            ext = os.path.splitext(image_path)[1].lower().lstrip('.')
            mime_map = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "gif": "image/gif",
                "webp": "image/webp",
            }
            mime = mime_map.get(ext, "image/jpeg")

            response = self.client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback: try with smaller vision model
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime};base64,{encoded_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e2:
                return f"Image processing failed: {e2}"
