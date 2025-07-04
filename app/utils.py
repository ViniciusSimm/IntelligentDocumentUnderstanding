from PIL import Image, ImageOps
import numpy as np
import cv2

def preprocess_pil_image(image: Image.Image) -> Image.Image:
    # 1. Converter para escala de cinza
    gray = image.convert('L')
    
    # 2. Aumentar contraste suavemente
    contrasted = ImageOps.autocontrast(gray, cutoff=2)
    
    # 3. Converter para numpy array
    np_img = np.array(contrasted)
    
    # 4. Binarização global simples (Otsu)
    _, binarized = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. (Opcional) Pequena morfologia para limpar pontos (não dilatar!)
    kernel = np.ones((1,1), np.uint8)
    cleaned = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)
    
    # 6. Converter de volta para PIL Image
    processed_image = Image.fromarray(cleaned)
    return processed_image
