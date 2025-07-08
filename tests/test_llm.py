
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from app.llm import load_index, search_similar_docs, build_prompt, generate_with_qwen, classify_text

class TestLlm(unittest.TestCase):

    @patch('app.llm.faiss.read_index')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"key": "value"}')
    def test_load_index(self, mock_open, mock_read_index):
        # Mock the return value of faiss.read_index
        mock_read_index.return_value = MagicMock()

        # Test the function
        index, labels, docs = load_index()
        self.assertIsNotNone(index)
        self.assertEqual(labels, {'key': 'value'})
        self.assertEqual(docs, {'key': 'value'})

    @patch('app.llm.load_index')
    @patch('app.llm.SentenceTransformer')
    def test_search_similar_docs(self, mock_sentence_transformer, mock_load_index):
        # Mock the return value of load_index
        mock_index = MagicMock()
        mock_index.search.return_value = (np.array([[]]), np.array([[]]))
        mock_load_index.return_value = (mock_index, ['label1', 'label2'], [{'text': 'doc1'}, {'text': 'doc2'}])
        # Mock the return value of SentenceTransformer
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2]]
        mock_sentence_transformer.return_value = mock_model

        # Test the function
        results = search_similar_docs('test query')
        self.assertEqual(len(results), 0)

    def test_build_prompt(self):
        # Test the function
        context_docs = [{'text': 'doc1', 'label': 'label1'}]
        query_text = 'test query'
        prompt = build_prompt(context_docs, query_text)
        self.assertIn('doc1', prompt)
        self.assertIn('label1', prompt)
        self.assertIn('test query', prompt)

    @patch('app.llm.requests.post')
    def test_generate_with_qwen(self, mock_post):
        # Mock the return value of requests.post
        mock_response = MagicMock()
        mock_response.json.return_value = {'response': 'test response'}
        mock_post.return_value = mock_response

        # Test the function
        response = generate_with_qwen('test prompt')
        self.assertEqual(response, 'test response')

    @patch('app.llm.search_similar_docs')
    @patch('app.llm.build_prompt')
    @patch('app.llm.generate_with_qwen')
    @patch('app.llm.extract_json_from_response')
    def test_classify_text(self, mock_extract_json, mock_generate, mock_build_prompt, mock_search_similar_docs):
        # Mock the return values of the patched functions
        mock_search_similar_docs.return_value = []
        mock_build_prompt.return_value = 'test prompt'
        mock_generate.return_value = 'test response'
        mock_extract_json.return_value = {'key': 'value'}

        # Test the function
        result = classify_text('test text')
        self.assertEqual(result['key'], 'value')
        self.assertIn('processing_time', result)

if __name__ == '__main__':
    unittest.main()
