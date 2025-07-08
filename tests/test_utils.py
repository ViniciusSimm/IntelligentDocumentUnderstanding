
import unittest
import numpy as np
from PIL import Image
from app.utils import binarize_image, remove_surrounded_black_pixels, preprocess_for_ocr, extract_json_from_response

class TestUtils(unittest.TestCase):

    def test_binarize_image(self):
        # Create a simple grayscale image
        img = np.array([[100, 200], [50, 150]], dtype=np.uint8)
        binary_img = binarize_image(img)
        # Check that the output is a binary image
        self.assertTrue(np.all(np.logical_or(binary_img == 0, binary_img == 255)))

    def test_remove_surrounded_black_pixels(self):
        # Create an image with a single black pixel surrounded by white pixels
        img = np.full((10, 10), 255, dtype=np.uint8)
        img[5, 5] = 0
        denoised_img = remove_surrounded_black_pixels(img)
        # Check that the black pixel has been removed
        self.assertEqual(denoised_img[5, 5], 255)

    def test_preprocess_for_ocr(self):
        # Create a simple RGB image
        img = Image.new('RGB', (100, 100), color = 'red')
        processed_img = preprocess_for_ocr(img)
        # Check that the output is a numpy array
        self.assertIsInstance(processed_img, np.ndarray)

    def test_extract_json_from_response(self):
        # Test with a valid JSON string
        response = '<think>Some thinking...</think>{"key": "value"}'
        json_obj = extract_json_from_response(response)
        self.assertEqual(json_obj, {"key": "value"})

        # Test with an invalid JSON string
        response = '<think>Some thinking...</think>{"key": "value"'
        json_obj = extract_json_from_response(response)
        self.assertEqual(json_obj, {})

        # Test with no JSON string
        response = '<think>Some thinking...</think>'
        json_obj = extract_json_from_response(response)
        self.assertEqual(json_obj, {})

if __name__ == '__main__':
    unittest.main()
