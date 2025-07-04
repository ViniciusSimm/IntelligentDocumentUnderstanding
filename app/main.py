from fastapi import FastAPI, File, UploadFile
from app.ocr import extract_text_from_upload_file
from app.llm import classify_text

app = FastAPI()

@app.post("/extract_text/")
async def extract_text(file: UploadFile = File(...)):
    content = await file.read()
    text = extract_text_from_upload_file(content, file.content_type)

    classification = classify_text(text)

    return {"class": classification, "text": text}

