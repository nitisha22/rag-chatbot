
from pathlib import Path
import re

from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent
PDF_FILE = BASE_DIR / "documents" / "sample.pdf"
OUTPUT_FILE = BASE_DIR / "output" / "extracted_text.txt"


def clean_page_text(text):
    """Normalize whitespace in extracted PDF text."""
    if not text:
        return ""

    # Join words separated by repeated spaces or line breaks.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():
    if not PDF_FILE.exists():
        print(f"Error: PDF not found: {PDF_FILE}")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(PDF_FILE))
    extracted_pages = []
    total_characters = 0

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_text = clean_page_text(raw_text)

        extracted_pages.append(
            f"--- Page {page_number} ---\n{cleaned_text}"
        )
        total_characters += len(cleaned_text)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write("\n\n".join(extracted_pages))

    print(f"Pages processed: {len(reader.pages)}")
    print(f"Characters extracted: {total_characters}")
    print(f"Saved extracted text to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
