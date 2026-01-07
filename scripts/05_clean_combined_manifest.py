from __future__ import annotations
from pathlib import Path
import pandas as pd

INP = Path("data/fma_manifest_combined.csv")
OUT = Path("data/fma_manifest_combined_clean.csv")
OUT_TEXT_ONLY = Path("data/fma_manifest_combined_text_only_clean.csv")

def to_empty_if_nan(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s


def main():
    if not INP.exists():
        raise FileNotFoundError(f"Missing input manifest: {INP}")

    df = pd.read_csv(INP)

    # Clean common columns if present
    for col in [
        "lyrics_path",
        "lyrics_source",
        "lyrics_path_genius",
        "lyrics_path_whisper",
        "lyrics_path_api",
        "lyrics_source_api",
        "lyrics_path_whisper",
        "lyrics_source_whisper",
        "text_path_combined",
        "text_source_combined",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(to_empty_if_nan)

    # If text_path_combined not present, stop with a clear message
    if "text_path_combined" not in df.columns:
        raise ValueError(
            "Column 'text_path_combined' not found in manifest. "
            "Run the combine script first to generate it."
        )

    # Check whether combined text file actually exists on disk
    df["text_exists"] = df["text_path_combined"].apply(
        lambda p: bool(p) and Path(p).exists()
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Write full cleaned manifest
    df.to_csv(OUT, index=False)

    # Write text-only cleaned manifest
    df_text = df[df["text_exists"]].copy()
    df_text.to_csv(OUT_TEXT_ONLY, index=False)

    print("Cleaned manifest written:", OUT)
    print("Cleaned text-only manifest written:", OUT_TEXT_ONLY)
    print(f"Text exists: {df_text['text_exists'].sum()} / {len(df)}")


if __name__ == "__main__":
    main()
