
from pathlib import Path
import json

from sentence_transformers import SentenceTransformer


def main():
    input_path = Path("week01/output/extracted_text.txt")
    output_path = Path("week01/output/embeddings.json")

    if not input_path.is_file():
        print(f"Input file not found: {input_path.resolve()}")
        return

    # Read the extracted text and split it into chunks.
    text = input_path.read_text(encoding="utf-8")

    chunk_size = 1000
    overlap = 150
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    print(f"Chunks to embed: {len(chunks)}")

    # Load a pretrained embedding model.
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading model: {model_name}")

    model = SentenceTransformer(model_name)

    # Convert each chunk into a numerical vector.
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Save each chunk alongside its embedding.
    records = []

    for index, (chunk, vector) in enumerate(
        zip(chunks, embeddings), start=1
    ):
        records.append({
            "chunk_id": index,
            "text": chunk,
            "embedding": vector.tolist()
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nEmbedding generation complete!")
    print(f"Chunks embedded: {len(embeddings)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Embedding array shape: {embeddings.shape}")
    print(f"Saved to: {output_path.resolve()}")

    print("\nFirst embedding preview:")
    print(embeddings[0][:10])


if __name__ == "__main__":
    main()
