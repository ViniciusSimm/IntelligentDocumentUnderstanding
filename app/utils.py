import cv2
import re
import json
import numpy as np
from PIL import Image
from scipy.ndimage import generic_filter

def binarize_image(gray: np.ndarray) -> np.ndarray:
    """
    Binarize image
    
    Args:
        gray (np.ndarray): Gray image

    Returns:
        np.ndarray: Binarized image.
    """

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def remove_surrounded_black_pixels(image: np.ndarray) -> np.ndarray:
    """
    Check nearby pixels to eliminate outliers

    Args:
        image (np.ndarray): Image with noise
    
    Returns:
        np.ndarray: Image without isolated black pixels.
    """

    def filtro(neigh):
        center = neigh[len(neigh) // 2]
        if center == 0:
            vizinhos_sem_centro = np.delete(neigh, len(neigh) // 2)
            if np.all(vizinhos_sem_centro == 255):
                return 255
        return center

    filtrada = generic_filter(image, filtro, size=7, mode='constant', cval=255)
    return filtrada.astype(np.uint8)

def preprocess_for_ocr(pil_image: Image.Image) -> np.ndarray:
    """
    Apply simple preprocessing for images

    Args:
        pil_image (np.ndarray): Image provided without treatment

    Returns:
        np.ndarray: Image with simple processing applied.
    """

    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = binarize_image(gray)
    denoised = remove_surrounded_black_pixels(binary)

    return denoised

def extract_json_from_response(response: str) -> dict:
    """
    Remove thinking strings

    Args:
        response (str): Answer from the LLM model.
    
    Returns:
        dict: JSON from the original answer.
    """

    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # Busca o primeiro bloco JSON
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            print("Invalid JSON")
            return {}
    else:
        print("No JSON found")
        return {}