import os
import whisper
import shutil
import argparse
import sys
from pathlib import Path as _Path
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = whisper.load_model("turbo")  # Whisper model

# Input and output file paths
MANIFEST_IN = Path("data/fma_manifest_3k_6genres_lyrics.csv")
MANIFEST_OUT = Path("data/fma_manifest_3k_6genres_lyrics_whisper.csv")

# Audio and output directories
AUDIO_DIR = Path("data/fma_small/fma_small")
TRANSCRIPTIONS_DIR = Path("data/whisper_transcriptions")  # For Whisper transcriptions

TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

#checks
parser = argparse.ArgumentParser(description="Transcribe FMA audio with Whisper")
parser.add_argument("--dry-run", action="store_true", help="Scan for files and report missing audio, then exit")
parser.add_argument("--limit", type=int, default=None, help="Limit number of manifest rows to process (for testing)")
args = parser.parse_args()

def _try_add_win_get_ffmpeg_to_path():
    # Look for WinGet-installed Gyan.FFmpeg package under the current user's WinGet packages
    try:
        winget_packages = _Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
        if winget_packages.exists():
            for p in winget_packages.iterdir():
                if p.name.lower().startswith("gyan.ffmpeg_"):
                    for f in p.rglob("ffmpeg.exe"):
                        ff_dir = str(f.parent)
                        os.environ["PATH"] = ff_dir + os.pathsep + os.environ.get("PATH", "")
                        return True
    except Exception:
        return False
    return False

if shutil.which("ffmpeg") is None:
    if _try_add_win_get_ffmpeg_to_path():
        print("ffmpeg discovered in WinGet packages and added to PATH.")
    else:
        print("ERROR: ffmpeg not found on PATH. Install ffmpeg and restart your terminal.")
        sys.exit(1)

print("ffmpeg found at:", shutil.which("ffmpeg"))

print("Loading manifest...")
df = pd.read_csv(MANIFEST_IN)

# Filter out songs that already have lyrics from Genius
df["lyrics_source"] = df["lyrics_source"].fillna("")
df_filtered = df[df["lyrics_source"].str.lower() != "genius"]

#scan disk
print(f"Scanning {AUDIO_DIR} for mp3 files...")
mp3_map = {}  # Dictionary to map {track_id (int) : full_file_path (Path)}

for root, dirs, files in os.walk(AUDIO_DIR):
    for file in files:
        if file.endswith(".mp3"):
            try:
                track_id_from_file = int(file.split('.')[0])
                full_path = Path(root) / file
                mp3_map[track_id_from_file] = full_path
            except ValueError:
                continue

print(f"Found {len(mp3_map)} audio files on disk.")

#dry-run requested, find missing audio, exit
if args.dry_run:
    print("\n--- Dry run: comparing manifest to found audio files ---")
    missing = []
    to_check = df_filtered if args.limit is None else df_filtered.head(args.limit)
    for idx, row in to_check.iterrows():
        track_id = int(row["track_id"])
        if track_id not in mp3_map:
            missing.append((track_id, row.get("artist", ""), row.get("title", "")))

    print(f"Manifest rows checked: {len(to_check)}")
    print(f"Audio files found on disk: {len(mp3_map)}")
    print(f"Missing audio files for {len(missing)} manifest entries (showing up to 20):")
    for t in missing[:20]:
        print(f" - {t[0]}: {t[1]} - {t[2]}")
    print("\nDry run complete. No transcription performed.")
    sys.exit(0)

processed_songs = []

# Function to transcribe
def transcribe_audio(audio_path: Path) -> str:
    # Convert Path object to string for Whisper
    audio_path_str = str(audio_path.resolve())
    
    # Load and transcribe
    audio = whisper.load_audio(audio_path_str)
    audio = whisper.pad_or_trim(audio)
    result = model.transcribe(audio)
    return result["text"]

# Iterate through the filtered songs
print("Starting transcription...")
for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Transcribing"):
    artist = row["artist"]
    title = row["title"]
    track_id = int(row["track_id"])

    # Construct the transcription filename using artist, title, and track_id
    transcription_filename = f"{artist} - {title} {track_id}.txt"
    
    # Construct the path to the mp3 file using subdirectories
    track_id_str = f"{track_id:06d}"
    audio_file = mp3_map.get(track_id, None)

    # Print the file path for debugging
    print(f"Checking file: {audio_file}")

    # CHECK IF WE ACTUALLY HAVE THE FILE
    if audio_file:
        # Check if already processed in this run
        try:
            # Transcribe audio using Whisper
            lyrics_text = transcribe_audio(audio_file)

            # Save transcription with the custom filename
            transcription_file = TRANSCRIPTIONS_DIR / transcription_filename
            with open(transcription_file, "w", encoding="utf-8") as f:
                f.write(lyrics_text)

            # Update DataFrame with the new transcription file path and source
            df.at[idx, "lyrics_path"] = str(transcription_file)
            df.at[idx, "lyrics_source"] = "whisper"
            processed_songs.append(track_id)

            # Print confirmation
            print(f"Transcription for {artist} - {title} saved!")

        except Exception as e:
            print(f"Error processing {artist} - {title}: {str(e)}")

    else:
        print(f"Audio file not found for {artist} - {title} at {audio_file}")

    # Add delay to avoid hitting rate limits (optional)
    time.sleep(1)

df.to_csv(MANIFEST_OUT, index=False)
print(f"\nDone. Processed {len(processed_songs)} songs.")
print(f"Updated manifest saved to: {MANIFEST_OUT}")
