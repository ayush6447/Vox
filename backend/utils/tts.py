"""
Text-to-Speech utility for SignSpeak.

Uses gTTS to synthesize speech from text and returns a temporary
audio file path that the FastAPI endpoint can stream.
"""

import tempfile
from pathlib import Path
from typing import Tuple

from gtts import gTTS


def synthesize_speech(text: str, lang: str = "en") -> Tuple[Path, str]:
    """
    Convert text to speech audio using gTTS.

    Returns:
        (path_to_file, mime_type)
    """
    # Create a temp file with .mp3 suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_path = Path(tmp.name)
    tmp.close()

    tts = gTTS(text=text, lang=lang)
    tts.save(str(tmp_path))

    return tmp_path, "audio/mpeg"



