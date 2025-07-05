"""
Unit tests for utility functions.
"""

import pytest
from ai_utils.utils.text_processing import (
    clean_text, normalize_text, chunk_text, 
    extract_sentences, remove_duplicates
)

class TestTextProcessing:
    """Test text processing utilities"""
    
    def test_clean_text(self):
        """Test text cleaning"""
        # Test basic cleaning
        assert clean_text("  Hello   World  ") == "Hello World"
        
        # Test special character removal
        assert clean_text("Hello@#$%World") == "HelloWorld"
        
        # Test empty text
        assert clean_text("") == ""
        assert clean_text(None) == ""
    
    def test_normalize_text(self):
        """Test text normalization"""
        # Test lowercase conversion
        assert normalize_text("Hello WORLD") == "hello world"
        
        # Test whitespace normalization
        assert normalize_text("  Hello   World  ") == "hello world"
        
        # Test empty text
        assert normalize_text("") == ""
        assert normalize_text(None) == ""
    
    def test_chunk_text(self):
        """Test text chunking"""
        text = "This is a sample text for testing chunking functionality"
        
        # Test with default parameters
        chunks = chunk_text(text)
        assert len(chunks) > 0
        assert all(len(chunk.split()) <= 512 for chunk in chunks)  # Default chunk size
        
        # Test with custom parameters
        chunks = chunk_text(text, chunk_size=5, chunk_overlap=1)
        assert len(chunks) > 0
        assert all(len(chunk.split()) <= 5 for chunk in chunks)
        
        # Test empty text
        assert chunk_text("") == []
        assert chunk_text(None) == []
    
    def test_extract_sentences(self):
        """Test sentence extraction"""
        text = "This is sentence one. This is sentence two! This is sentence three?"
        sentences = extract_sentences(text)
        
        assert len(sentences) == 3
        assert "This is sentence one" in sentences[0]
        assert "This is sentence two" in sentences[1]
        assert "This is sentence three" in sentences[2]
        
        # Test empty text
        assert extract_sentences("") == []
        assert extract_sentences(None) == []
    
    def test_remove_duplicates(self):
        """Test duplicate removal"""
        texts = [
            "Hello World",
            "hello world",  # Duplicate after normalization
            "Test Text",
            "test text",    # Duplicate after normalization
            "Unique Content"
        ]
        
        result = remove_duplicates(texts)
        assert len(result) == 3  # Should remove 2 duplicates
        assert "Hello World" in result
        assert "Test Text" in result
        assert "Unique Content" in result
        
        # Test empty list
        assert remove_duplicates([]) == []
        assert remove_duplicates(None) == [] 