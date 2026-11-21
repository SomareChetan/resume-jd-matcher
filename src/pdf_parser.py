"""
PDF parser module using PyMuPDF (fitz).
Extracts raw text from PDF files and handles unreadable/empty PDFs gracefully.
"""

from typing import Tuple
import fitz  # PyMuPDF


def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, str]:
    """
    Extracts text from PDF bytes.

    Args:
        file_bytes (bytes): Raw bytes of the uploaded PDF file.

    Returns:
        Tuple[str, str]: (extracted_text, error_message).
                         If successful, error_message is empty.
    """
    if not file_bytes:
        return "", "The uploaded file is empty."

    try:
        # Open PDF document from stream
        doc = fitz.open(stream=file_bytes, filetype="pdf")

        if doc.is_encrypted:
            return "", "The PDF is password protected and cannot be read."

        if doc.page_count == 0:
            return "", "The PDF contains no pages."

        extracted_pages = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text:
                extracted_pages.append(text.strip())

        full_text = "\n\n".join(extracted_pages).strip()

        if not full_text:
            return "", (
                "No readable text could be extracted from this PDF. "
                "It may be a scanned image or contain non-selectable text. "
                "Please upload a text-based PDF."
            )

        return full_text, ""

    except Exception as e:
        return "", f"Failed to parse PDF file: {str(e)}"