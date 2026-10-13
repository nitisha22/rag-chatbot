
from pathlib import Path


def split_text(text: str, chunk_size: int = 1000,
               overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

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

    return chunks


def main():
    input_path = Path("week01/output/extracted_text.txt")
    output_path = Path("week01/output/chunks.txt")

    if not input_path.is_file():
        print(f"Input file not found: {input_path.resolve()}")
        return

    text = input_path.read_text(encoding="utf-8")
    chunks = split_text(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for index, chunk in enumerate(chunks, start=1):
            file.write(f"--- Chunk {index} ---\n")
            file.write(chunk)
            file.write("\n\n")

    print(f"Characters read: {len(text)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Chunk size: 1000 characters")
    print(f"Overlap: 150 characters")
    print(f"Saved chunks to: {output_path.resolve()}")

    if chunks:
        print("\nFirst chunk preview:\n")
        print(chunks[0][:500])


if __name__ == "__main__":
    main()