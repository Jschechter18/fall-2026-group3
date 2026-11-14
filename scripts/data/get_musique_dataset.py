"""Download MuSiQue (``dgslibisey/MuSiQue`` on the Hub) into ``data/MuSiQue/clean/``.

Writes ``train.json`` and ``validation.json``, which hold JSON Lines despite the suffix; the
distributed collection configs read them as ``train.jsonl`` / ``validation.jsonl`` copies.
"""
from pathlib import Path

from mas_sae.data.musique import download_musique

if __name__ == "__main__":
    download_musique(Path("data/MuSiQue/clean"))    
