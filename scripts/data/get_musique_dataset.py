from pathlib import Path

from mas_sae.data.musique import download_musique

if __name__ == "__main__":
    download_musique(Path("data/MuSiQue/clean"))    
