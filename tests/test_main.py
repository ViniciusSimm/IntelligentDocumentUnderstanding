
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

class TestMain(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    @patch('app.main.extract_text_from_upload_file')
    @patch('app.main.classify_text')
    def test_extract_text(self, mock_classify_text, mock_extract_text):
        # Mock the return values of the patched functions
        mock_extract_text.return_value = 'test text'
        mock_classify_text.return_value = {'document_type': 'test_type'}

        # Create a dummy file
        dummy_file = ('test.jpg', b'dummy content', 'image/jpeg')

        # Test the endpoint
        response = self.client.post("/extract_text/", files={"file": dummy_file})

        # Assert the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'document_type': 'test_type'})

if __name__ == '__main__':
    unittest.main()
