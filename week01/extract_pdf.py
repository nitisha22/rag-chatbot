
from pathlib import Path
import sys

from pypdf import PdfReader


def extract_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF while preserving page numbers."""
    reader = PdfReader(str(pdf_path))
    extracted_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        extracted_pages.append(
            f"--- Page {page_number} ---\n{text}"
        )

    return "\n\n".join(extracted_pages)


def main():
    if len(sys.argv) != 2:
        print(
            'Usage: python week01\\extract_pdf.py '
            'week01\\documents\\sample.pdf'
        )
        return

    pdf_path = Path(sys.argv[1])

    if not pdf_path.is_file():
        print(f"Error: PDF not found: {pdf_path.resolve()}")
        return

    try:
        text = extract_pdf(pdf_path)

        output_dir = Path("week01") / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "extracted_text.txt"
        output_path.write_text(text, encoding="utf-8")

        print(f"Pages processed: {len(PdfReader(str(pdf_path)).pages)}")
        print(f"Characters extracted: {len(text)}")
        print(f"Saved extracted text to: {output_path.resolve()}")

        print("\nPreview:\n")
        print(text[:1000] or "No extractable text found.")

    except Exception as error:
        print(f"Could not process PDF: {error}")


if __name__ == "__main__":
    main()