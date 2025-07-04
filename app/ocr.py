from io import BytesIO
from typing import Literal
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
from pathlib import Path
import pytesseract
import json
from .utils import preprocess_pil_image

def extract_text_from_upload_file(file_bytes: bytes, content_type: Literal["application/pdf", "image/jpeg", "image/png", "image/tiff"]) -> str:
    """
    Extract text from uploaded file (image or PDF) using Tesseract OCR.

    Args:
        file_bytes (bytes): Raw file content.
        content_type (str): MIME type of the file.

    Returns:
        str: Extracted text.
    """

    custom_config = r'--oem 3 --psm 6'

    if content_type == "application/pdf":
        # Convert PDF to images and extract text from each page
        images = convert_from_bytes(file_bytes)
        texts = []
        for img in images:
            text = pytesseract.image_to_string(img, config=custom_config)
            texts.append(text.strip())
        return "\n\n".join(texts)

    elif content_type.startswith("image/"):
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        text = pytesseract.image_to_string(image)
        return text.strip()

    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def extract_documents(image_root="images", output_path="data/extracted_docs.json"):
    """
    Extract text from images in the specified directory and save as a JSON file.

    Args:
        image_root (str): Directory containing images organized in subfolders.
        output_path (str): Path to save the extracted documents as JSON.
    
    Returns:
        list: List of dictionaries containing extracted text, labels, and file paths.
    """
    
    image_root = Path(image_root)
    documents = []

    for folder in image_root.iterdir():
        print(folder)
        if folder.is_dir():
            for image_path in folder.glob("*"):
                try:
                    image = Image.open(image_path)
                    text = pytesseract.image_to_string(image, lang="eng")
                    documents.append({
                        "text": text.strip(),
                        "label": folder.name,
                        "file": str(image_path)
                    })
                except Exception as e:
                    print(f"[ERROR] {image_path}: {e}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    return documents

