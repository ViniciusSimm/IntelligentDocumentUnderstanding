
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
from app.ocr import extract_text_from_upload_file, extract_documents

class TestOcr(unittest.TestCase):

    @patch('app.ocr.convert_from_bytes')
    @patch('app.ocr.pytesseract.image_to_string')
    def test_extract_text_from_pdf(self, mock_image_to_string, mock_convert_from_bytes):
        # Mock the return value of convert_from_bytes
        mock_convert_from_bytes.return_value = [Image.new('RGB', (100, 100))]
        # Mock the return value of image_to_string
        mock_image_to_string.return_value = 'test text'

        # Test with a PDF file
        file_bytes = b'pdf content'
        content_type = 'application/pdf'
        text = extract_text_from_upload_file(file_bytes, content_type, False)
        self.assertEqual(text, 'test text')

    @patch('app.ocr.Image.open')
    @patch('app.ocr.pytesseract.image_to_string')
    def test_extract_text_from_image(self, mock_image_to_string, mock_image_open):
        # Mock the return value of Image.open
        mock_image_open.return_value = Image.new('RGB', (100, 100))
        # Mock the return value of image_to_string
        mock_image_to_string.return_value = 'test text'

        # Test with an image file
        file_bytes = b'image content'
        content_type = 'image/jpeg'
        text = extract_text_from_upload_file(file_bytes, content_type, False)
        self.assertEqual(text, 'test text')

    @patch('app.ocr.Path.iterdir')
    @patch('app.ocr.Image.open')
    @patch('app.ocr.pytesseract.image_to_string')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_extract_documents(self, mock_open, mock_image_to_string, mock_image_open, mock_iterdir):
        # Mock the directory structure
        mock_folder = MagicMock()
        mock_folder.is_dir.return_value = True
        mock_folder.name = 'test_folder'
        mock_image_path = MagicMock()
        mock_image_path.glob.return_value = [MagicMock()]
        mock_folder.glob.return_value = [mock_image_path]
        mock_iterdir.return_value = [mock_folder]

        # Mock the return value of Image.open and image_to_string
        mock_image_open.return_value = Image.new('RGB', (100, 100))
        mock_image_to_string.return_value = 'test text'

        # Test the function
        documents = extract_documents()
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]['text'], 'test text')
        self.assertEqual(documents[0]['label'], 'test_folder')

if __name__ == '__main__':
    unittest.main()
