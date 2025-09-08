# app/config.py
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    piper_tts_file_path: str = os.getenv("PIPER_TTS_MODEL", "")

settings = Settings()
