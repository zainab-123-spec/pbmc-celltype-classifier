"""Download the public PBMC3k counts from 10x Genomics (7.6 MB) into data/raw/."""

from scpbmc.data import download_pbmc3k

if __name__ == "__main__":
    print("PBMC3k matrix at:", download_pbmc3k("data/raw"))
    print("PBMC68k-reduced ships inside scanpy; nothing to download.")
