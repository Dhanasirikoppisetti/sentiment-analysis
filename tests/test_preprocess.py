"""
Unit tests for text preprocessing functions.
"""

import pytest
from app.utils.preprocess import clean_text, batch_clean_texts


class TestCleanText:
    """Tests for the clean_text function."""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "Hello World!"
        result = clean_text(text)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_clean_text_lowercase(self):
        """Test that text is converted to lowercase."""
        text = "HELLO WORLD"
        result = clean_text(text)
        assert result == result.lower()
    
    def test_clean_text_removes_html(self):
        """Test that HTML tags are removed."""
        text = "<p>This is a <b>test</b></p>"
        result = clean_text(text)
        assert "<" not in result
        assert ">" not in result
    
    def test_clean_text_removes_numbers(self):
        """Test that numbers are removed."""
        text = "I have 123 apples and 456 oranges"
        result = clean_text(text)
        assert "123" not in result
        assert "456" not in result
    
    def test_clean_text_removes_punctuation(self):
        """Test that punctuation is removed."""
        text = "Hello! How are you? I'm great!"
        result = clean_text(text)
        for char in "!?'":
            assert char not in result
    
    def test_clean_text_removes_urls(self):
        """Test that URLs are removed."""
        text = "Check out https://example.com for more info"
        result = clean_text(text)
        assert "https" not in result
        assert "example.com" not in result
    
    def test_clean_text_removes_email(self):
        """Test that email addresses are removed."""
        text = "Contact me at test@example.com for details"
        result = clean_text(text)
        assert "@" not in result
        assert "example" not in result
    
    def test_clean_text_empty_string(self):
        """Test handling of empty string."""
        text = ""
        result = clean_text(text)
        assert result == ""
    
    def test_clean_text_whitespace_only(self):
        """Test handling of whitespace-only string."""
        text = "   \n\t  "
        result = clean_text(text)
        assert result == ""
    
    def test_clean_text_single_character(self):
        """Test handling of single character (should be removed as stopword)."""
        text = "a"
        result = clean_text(text)
        # Single characters are removed by the len(word) > 2 filter
        assert len(result) == 0
    
    def test_clean_text_multiple_spaces(self):
        """Test that multiple spaces are normalized to single space."""
        text = "hello    world    how    are    you"
        result = clean_text(text)
        assert "    " not in result
    
    def test_clean_text_sentiment_review(self):
        """Test cleaning of a realistic sentiment review."""
        text = "This movie is AMAZING!!! I love it so much. 10/10 stars!!!"
        result = clean_text(text)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "!" not in result
        assert "10" not in result
    
    def test_clean_text_non_string_input(self):
        """Test handling of non-string input."""
        result = clean_text(123)
        assert isinstance(result, str)
    
    def test_clean_text_special_characters(self):
        """Test removal of special characters."""
        text = "Hello@#$%World^&*!"
        result = clean_text(text)
        for char in "@#$%^&*!":
            assert char not in result


class TestBatchCleanTexts:
    """Tests for the batch_clean_texts function."""
    
    def test_batch_clean_empty_list(self):
        """Test batch cleaning with empty list."""
        result = batch_clean_texts([])
        assert result == []
    
    def test_batch_clean_single_text(self):
        """Test batch cleaning with single text."""
        texts = ["Hello World!"]
        result = batch_clean_texts(texts)
        assert len(result) == 1
        assert isinstance(result[0], str)
    
    def test_batch_clean_multiple_texts(self):
        """Test batch cleaning with multiple texts."""
        texts = [
            "This is GREAT!",
            "This is TERRIBLE!",
            "This is OKAY."
        ]
        result = batch_clean_texts(texts)
        assert len(result) == 3
        assert all(isinstance(text, str) for text in result)
    
    def test_batch_clean_consistency(self):
        """Test that batch cleaning produces same result as individual cleaning."""
        texts = ["Hello! World?", "Test@Example.COM"]
        batch_result = batch_clean_texts(texts)
        individual_result = [clean_text(text) for text in texts]
        assert batch_result == individual_result
    
    def test_batch_clean_with_mixed_content(self):
        """Test batch cleaning with various content types."""
        texts = [
            "<p>HTML content with numbers 123</p>",
            "Pure text without special chars",
            "Email@test.com and URL https://example.com"
        ]
        result = batch_clean_texts(texts)
        assert len(result) == 3
        assert all(len(text) > 0 for text in result if text)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    def test_very_long_text(self):
        """Test cleaning of very long text."""
        text = "word " * 1000
        result = clean_text(text)
        assert isinstance(result, str)
    
    def test_only_stopwords(self):
        """Test text containing only stopwords."""
        text = "a an the is are"
        result = clean_text(text)
        # Most of these are stopwords, result should be empty or very short
        assert isinstance(result, str)
    
    def test_mixed_case_and_special(self):
        """Test mixed case with special characters."""
        text = "ThIs Is A MixED cAsE tExT!!! 123"
        result = clean_text(text)
        assert result == result.lower()
        assert "!" not in result
        assert "123" not in result
    
    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        text = "Café, naïve, résumé"
        result = clean_text(text)
        assert isinstance(result, str)
        # Accented characters should be preserved in lowercase
    
    def test_multiple_spaces_and_newlines(self):
        """Test text with multiple spaces and newlines."""
        text = "Hello   \n\n   World   \t\t  Test"
        result = clean_text(text)
        assert "   " not in result
        assert "\n" not in result
        assert "\t" not in result
