
import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "output" / "extracted_text.txt"
OUTPUT_FILE = BASE_DIR / "output" / "embeddings.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def split_into_page_chunks(text):
    """Split extracted text into chunks without crossing page boundaries."""

    page_markers = list(re.finditer(r"--- Page (\d+) ---", text))
    chunks = []

    if not page_markers:
        raise ValueError(
            "No page markers found. Expected markers like --- Page 1 ---"
        )

    for i, marker in enumerate(page_markers):
        page_number = int(marker.group(1))
        page_start = marker.end()

        if i + 1 < len(page_markers):
            page_end = page_markers[i + 1].start()
        else:
            page_end = len(text)

        page_text = text[page_start:page_end].strip()

        start = 0

        while start < len(page_text):
            end = min(start + CHUNK_SIZE, len(page_text))
            chunk_text = page_text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "page_number": page_number,
                    "text": chunk_text,
                })

            if end == len(page_text):
                break

            start = end - CHUNK_OVERLAP

    return chunks


def main():
    if not INPUT_FILE.is_file():
        print(f"Input file not found: {INPUT_FILE}")
        return

    text = INPUT_FILE.read_text(encoding="utf-8")
    chunks = split_into_page_chunks(text)

    if not chunks:
        print("No chunks were created.")
        return

    print(f"Pages detected: {len(set(c['page_number'] for c in chunks))}")
    print(f"Chunks to embed: {len(chunks)}")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    chunk_texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        chunk_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    records = []

    for index, (chunk, vector) in enumerate(
        zip(chunks, embeddings), start=1
    ):
        records.append({
            "chunk_id": index,
            "page_number": chunk["page_number"],
            "text": chunk["text"],
            "embedding": vector.tolist(),
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(records, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nEmbedding generation complete!")
    print(f"Chunks embedded: {len(records)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Embedding array shape: {embeddings.shape}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nFirst record preview:")
    print(f"Chunk ID: {records[0]['chunk_id']}")
    print(f"Source page: {records[0]['page_number']}")
    print(f"Text preview: {records[0]['text'][:200]}")


if __name__ == "__main__":
    main()
