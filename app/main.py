from fastapi import FastAPI, File, UploadFile, Query
from app.ocr import extract_text_from_upload_file
from app.llm import classify_text

app = FastAPI()

@app.post("/extract_text/")
async def extract_text(
    file: UploadFile = File(...),
    apply_preprocessing: bool = Query(False, description="Apply image preprocessing?")
):
    content = await file.read()
    text = extract_text_from_upload_file(content, file.content_type, apply_preprocessing)

    classification = classify_text(text)

    return classification

