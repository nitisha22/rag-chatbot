
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# 1. Configure paths and model
BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_FILE = BASE_DIR / "output" / "embeddings.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10


def main():
    # 2. Load saved chunks and their embeddings
    if not EMBEDDINGS_FILE.exists():
        print(f"Error: Embeddings file not found:\n{EMBEDDINGS_FILE}")
        print("Run embed_chunks.py first.")
        return

    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    if not records:
        print("Error: No chunks found in embeddings.json.")
        return

    print(f"Loaded {len(records)} embedded chunks.")

    # 3. Exclude the table-of-contents chunk from retrieval
    original_count = len(records)

    records = [
        record
        for record in records
        if not (
            record.get("page_number") == 2
            and "CONTENTS" in record.get("text", "")[:100].upper()
        )
    ]

    print(f"Chunks before filtering: {original_count}")
    print(f"Chunks after filtering: {len(records)}")

    if not records:
        print("Error: No searchable chunks remain.")
        return

    # 4. Prepare the stored embedding vectors
    vectors = np.array(
        [record["embedding"] for record in records],
        dtype=np.float32,
    )

    # 5. Load the same model used to create the embeddings
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 6. Ask the user a question
    query = input("\nEnter your question: ").strip()

    if not query:
        print("Please enter a non-empty question.")
        return

    # 7. Convert the question into an embedding
    query_vector = model.encode(
        query,
        convert_to_numpy=True,
    ).astype(np.float32)

    # 8. Calculate cosine similarity
    vector_norms = np.linalg.norm(vectors, axis=1)
    query_norm = np.linalg.norm(query_vector)

    similarities = (vectors @ query_vector) / (
        vector_norms * query_norm + 1e-10
    )

    # 9. Find the top matching chunks
    top_indices = np.argsort(similarities)[::-1][:TOP_K]

    # 10. Display the results
    print("\n" + "=" * 70)
    print("TOP MATCHING CHUNKS")
    print("=" * 70)

    for rank, index in enumerate(top_indices, start=1):
        record = records[index]
        score = float(similarities[index])

        print(f"\nRank: {rank}")
        print(f"Chunk ID: {record.get('chunk_id', index)}")
        print(f"Source page: {record.get('page_number', 'Unknown')}")
        print(f"Cosine similarity: {score:.4f}")
        print("-" * 70)
        print(record["text"])
        print("-" * 70)


if __name__ == "__main__":
    main()
