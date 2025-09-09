# tts_service.py
import io
import wave, audioop
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np
import soundfile as sf
from piper import PiperVoice
from piper.config import SynthesisConfig
from config import settings
from pydub import AudioSegment
from contextlib import asynccontextmanager

voice = None
syn_config = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model during startup
    global voice, syn_config
    try:
        VOICE_MODEL_PATH = settings.piper_tts_file_path
        print(f"Loading voice model from: {VOICE_MODEL_PATH}")
        voice = PiperVoice.load(VOICE_MODEL_PATH, use_cuda=True)
        syn_config = SynthesisConfig(
            noise_scale=0.5,
            length_scale=1.05,
            noise_w_scale=0.8
        )
        print("Voice model loaded successfully!")
        yield
    except Exception as e:
        print(f"Failed to load voice model: {e}")
        import traceback
        traceback.print_exc()
        yield  # Still yield even if loading fails
    finally:
        # Cleanup code can go here
        pass


def wav_to_twilio_mulaw(wav_io):
    # Load WAV
    audio_segment = AudioSegment.from_wav(wav_io)
    audio_segment = audio_segment.set_frame_rate(8000).set_channels(1).set_sample_width(2)  # 16-bit PCM

    # Get raw PCM bytes
    pcm_data = audio_segment.raw_data

    # Convert PCM16 → μ-law (PCMU)
    mulaw_data = audioop.lin2ulaw(pcm_data, 2)  # 2 = sample width (16-bit)

    return mulaw_data  # raw bytes, no header

app = FastAPI(lifespan=lifespan)

# Load TTS model (using Microsoft's VITS model as example)
voice = None
syn_config = None


def text_to_speech(text: str):
    """Convert text to mu-law encoded audio bytes"""
    try:
        # Tokenize input text
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            voice.synthesize_wav(
                text,
                wav_file,
                syn_config=syn_config
            )
        wav_io.seek(0)

        # Step 2: Use pydub to load, resample, and convert to ulaw 8000 Hz
        ulaw_audio = wav_to_twilio_mulaw(wav_io)
        
        return ulaw_audio
    except Exception as e:
        print(f"Error in TTS conversion: {e}")
        return None


@app.websocket("/ws/tts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            text = data.get("text")
            request_id = data.get("contextId")
            
            if not text or not request_id:
                await websocket.send_json({
                    "error": "Missing text or id",
                    "id": request_id
                })
                continue
            
            print(f"Processing TTS for ID: {request_id}")
            
            # Convert text to speech
            audio_bytes = text_to_speech(text)
            
            if audio_bytes:
                # Encode audio bytes to base64 for JSON transmission
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                # Send back to client
                await websocket.send_json({
                    "contextId": request_id,
                    "audio": audio_b64,
                })
            else:
                await websocket.send_json({
                    "error": "TTS conversion failed",
                    "id": request_id
                })
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({
            "error": str(e),
            "id": request_id if 'request_id' in locals() else None
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)