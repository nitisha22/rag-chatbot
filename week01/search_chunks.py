
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# Build file paths relative to this Python file
BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_FILE = BASE_DIR / "output" / "embeddings.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


def main():
    # 1. Load the saved chunks and their embeddings
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

    # 2. Prepare the stored embedding vectors
    vectors = np.array(
        [record["embedding"] for record in records],
        dtype=np.float32,
    )

    # 3. Load the same model used to create the embeddings
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 4. Ask the user a question
    query = input("\nEnter your question: ").strip()

    if not query:
        print("Please enter a non-empty question.")
        return

    # 5. Convert the question into an embedding
    query_vector = model.encode(
        query,
        convert_to_numpy=True,
    ).astype(np.float32)

    # 6. Calculate cosine similarity between the query and each chunk
    vector_norms = np.linalg.norm(vectors, axis=1)
    query_norm = np.linalg.norm(query_vector)

    similarities = (vectors @ query_vector) / (
        vector_norms * query_norm + 1e-10
    )

    # 7. Find the three most similar chunks
    top_indices = np.argsort(similarities)[::-1][:TOP_K]

    # 8. Display the results
    print("\n" + "=" * 70)
    print("TOP MATCHING CHUNKS")
    print("=" * 70)

    for rank, index in enumerate(top_indices, start=1):
        record = records[index]
        score = float(similarities[index])

        print(f"\nRank: {rank}")
        print(f"Chunk ID: {record.get('chunk_id', index)}")
        print(f"Cosine similarity: {score:.4f}")
        print("-" * 70)
        print(record["text"])
        print("-" * 70)


if __name__ == "__main__":
    main()
