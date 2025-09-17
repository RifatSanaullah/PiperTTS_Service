import numpy as np
import librosa
from scipy import signal

import io
import wave
import audioop
import base64
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from piper import PiperVoice
from piper.config import SynthesisConfig
from config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

voice: PiperVoice = None
syn_config: SynthesisConfig = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the TTS model on application startup."""
    global voice, syn_config
    try:
        VOICE_MODEL_PATH = settings.piper_tts_file_path
        logging.info(f"Loading voice model from: {VOICE_MODEL_PATH}")
        voice = await asyncio.to_thread(PiperVoice.load, VOICE_MODEL_PATH, use_cuda=True)
        syn_config = SynthesisConfig(
            noise_scale=0.5,
            length_scale=1.0,
            noise_w_scale=0.6
        )
        logging.info("Voice model loaded successfully!")
        yield
    except Exception as e:
        logging.error(f"Failed to load voice model: {e}", exc_info=True)
        yield
    finally:
        # Cleanup can go here
        pass

# def wav_to_twilio_mulaw(wav_io):
#     """
#     Converts high-quality 22.05kHz WAV to 8kHz μ-law without echo.
#     Uses minimum-phase filter to avoid phase distortion.
#     """
#     try:
#         wav_io.seek(0)
#         y, orig_sr = librosa.load(wav_io, sr=22050, mono=True)
#         nyquist = orig_sr / 2.0
#         cutoff = 3600 / nyquist
        
#         b = signal.firwin(101, cutoff, window='hamming', pass_zero=True)
#         b_min = signal.minimum_phase(b)
        
#         y_filtered = signal.lfilter(b_min, 1.0, y)
#         y_resampled = librosa.resample(y_filtered, orig_sr=orig_sr, target_sr=8000, 
#                                      res_type='kaiser_fast')
        
#         y_resampled = y_resampled - np.mean(y_resampled)
        
#         peak = np.max(np.abs(y_resampled))
#         if peak > 0.9:
#             y_resampled = y_resampled * (0.99 / peak)
#         pcm_data = (y_resampled * 32767).astype(np.int16).tobytes()
#         mulaw_data = audioop.lin2ulaw(pcm_data, 2)
        
#         return mulaw_data
        
#     except Exception as e:
#         logging.error(f"Error in echo-free audio conversion: {e}", exc_info=True)
#         return None


def wav_to_twilio_mulaw(wav_io):
    """
    Converts high-quality 22.05kHz WAV to 8kHz μ-law with increased volume.
    """
    try:
        wav_io.seek(0)
        y, orig_sr = librosa.load(wav_io, sr=None, mono=True)
        gain_factor = 1.8
        y *= gain_factor
        nyquist = orig_sr / 2.0
        cutoff = 3600 / nyquist
        
        b = signal.firwin(101, cutoff, window='hamming', pass_zero=True)
        b_min = signal.minimum_phase(b)
        
        y_filtered = signal.lfilter(b_min, 1.0, y)
        y_resampled = librosa.resample(y_filtered, orig_sr=orig_sr, target_sr=8000, 
                                       res_type='soxr_hq') # Switched back to soxr_hq for better quality.
        y_resampled = y_resampled - np.mean(y_resampled)
        peak = np.max(np.abs(y_resampled))
        if peak > 1.0:
            y_resampled /= peak
        pcm_data = (y_resampled * 32767).astype(np.int16).tobytes()
        mulaw_data = audioop.lin2ulaw(pcm_data, 2)
        return mulaw_data
        
    except Exception as e:
        logging.error(f"Error in audio conversion: {e}", exc_info=True)
        return None

async def text_to_speech(text: str):
    """
    Converts text to μ-law encoded audio bytes using a background thread.
    """
    if not voice:
        logging.error("TTS model is not loaded.")
        return None
    try:
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            await asyncio.to_thread(voice.synthesize_wav, text, wav_file, syn_config=syn_config)
        wav_io.seek(0)
        ulaw_audio = await asyncio.to_thread(wav_to_twilio_mulaw, wav_io)
        return ulaw_audio
    except Exception as e:
        logging.error(f"Error in TTS conversion: {e}", exc_info=True)
        return None

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws/tts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connection established.")
    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("text")
            request_id = data.get("contextId")

            if not text or not request_id:
                logging.warning("Received invalid data: Missing text or id.")
                await websocket.send_json({"error": "Missing text or id", "contextId": request_id})
                continue
            
            logging.info(f"Processing TTS for ID: {request_id}, text: '{text[:30]}...'")
            
            audio_bytes = await text_to_speech(text)
            
            if audio_bytes:
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                await websocket.send_json({
                    "contextId": request_id,
                    "audio": audio_b64,
                })
                logging.info(f"Successfully sent audio for ID: {request_id}")
            else:
                await websocket.send_json({
                    "error": "TTS conversion failed",
                    "contextId": request_id
                })
                logging.error(f"TTS conversion failed for ID: {request_id}")
                
    except WebSocketDisconnect:
        logging.info("Client disconnected.")
    except Exception as e:
        logging.error(f"An unexpected WebSocket error occurred: {e}", exc_info=True)
        await websocket.send_json({
            "error": str(e),
            "contextId": request_id if 'request_id' in locals() else 'unknown'
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tts_service:app", host="0.0.0.0", port=8001, reload=True)