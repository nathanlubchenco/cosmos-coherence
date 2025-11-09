"""Tests for coherence integration and scoring."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from cosmos_coherence.benchmarks.implementations.coherence_integration import (
    CoherenceIntegration,
)


@pytest.fixture
def coherence_integration():
    """Create CoherenceIntegration instance with mocked coherence variants."""
    # Mock the selfcheckgpt imports to avoid requiring OpenAI API key in tests
    with patch('cosmos_coherence.benchmarks.implementations.coherence_integration.SelfCheckShogenji') as mock_shogenji, \
         patch('cosmos_coherence.benchmarks.implementations.coherence_integration.SelfCheckFitelson') as mock_fitelson, \
         patch('cosmos_coherence.benchmarks.implementations.coherence_integration.SelfCheckOlsson') as mock_olsson:

        # Create instance
        integration = CoherenceIntegration(model="gpt-4o-mini")

        # Store mocks for assertions
        integration._mock_shogenji_class = mock_shogenji
        integration._mock_fitelson_class = mock_fitelson
        integration._mock_olsson_class = mock_olsson

        return integration


class TestCoherenceIntegration:
    """Test CoherenceIntegration functionality."""

    def test_init_coherence_variant_shogenji(self, coherence_integration):
        """Test that Shogenji variant initializes correctly."""
        with patch.object(coherence_integration._mock_shogenji_class, 'return_value') as mock_instance:
            variant = coherence_integration._init_coherence_variant("shogenji")

            # Should have called the constructor
            coherence_integration._mock_shogenji_class.assert_called_once_with(model="gpt-4o-mini")

    def test_init_coherence_variant_fitelson(self, coherence_integration):
        """Test that Fitelson variant initializes correctly."""
        with patch.object(coherence_integration._mock_fitelson_class, 'return_value') as mock_instance:
            variant = coherence_integration._init_coherence_variant("fitelson")

            # Should have called the constructor
            coherence_integration._mock_fitelson_class.assert_called_once_with(model="gpt-4o-mini")

    def test_init_coherence_variant_olsson(self, coherence_integration):
        """Test that Olsson variant initializes correctly."""
        with patch.object(coherence_integration._mock_olsson_class, 'return_value') as mock_instance:
            variant = coherence_integration._init_coherence_variant("olsson")

            # Should have called the constructor
            coherence_integration._mock_olsson_class.assert_called_once_with(model="gpt-4o-mini")

    def test_init_coherence_variant_invalid(self, coherence_integration):
        """Test that invalid variant name raises clear error."""
        with pytest.raises(ValueError, match="Unknown coherence variant"):
            coherence_integration._init_coherence_variant("invalid_variant")

    def test_score_question_variations_shogenji(self, coherence_integration):
        """Test scoring question variations with Shogenji variant."""
        # Mock Shogenji instance
        mock_shogenji_instance = MagicMock()
        mock_shogenji_instance.predict.return_value = np.array([0.85])

        with patch.object(
            coherence_integration._mock_shogenji_class,
            'return_value',
            mock_shogenji_instance
        ):
            variant = coherence_integration._init_coherence_variant("shogenji")

            original_question = "What is the capital of France?"
            variations = [
                "Which city serves as France's capital?",
                "What city is the French capital?",
                "France's capital is located in which city?",
            ]

            score = coherence_integration.score_question_variations(
                variant, original_question, variations
            )

            # Should have called predict with correct arguments
            mock_shogenji_instance.predict.assert_called_once()
            call_args = mock_shogenji_instance.predict.call_args

            # Check sentences parameter (should be list with original question)
            assert call_args[1]['sentences'] == [original_question]
            # Check sampled_passages parameter (should be variations)
            assert call_args[1]['sampled_passages'] == variations

            # Should return a float score
            assert isinstance(score, float)
            assert score == 0.85

    def test_score_question_variations_fitelson(self, coherence_integration):
        """Test scoring question variations with Fitelson variant."""
        # Mock Fitelson instance
        mock_fitelson_instance = MagicMock()
        mock_fitelson_instance.predict.return_value = np.array([0.72])

        with patch.object(
            coherence_integration._mock_fitelson_class,
            'return_value',
            mock_fitelson_instance
        ):
            variant = coherence_integration._init_coherence_variant("fitelson")

            original_question = "What is the capital of France?"
            variations = [
                "Which city serves as France's capital?",
                "What city is the French capital?",
            ]

            score = coherence_integration.score_question_variations(
                variant, original_question, variations
            )

            # Should have called predict
            mock_fitelson_instance.predict.assert_called_once()

            # Should return a float score
            assert isinstance(score, float)
            assert score == 0.72

    def test_score_question_variations_olsson(self, coherence_integration):
        """Test scoring question variations with Olsson variant."""
        # Mock Olsson instance
        mock_olsson_instance = MagicMock()
        mock_olsson_instance.predict.return_value = np.array([0.68])

        with patch.object(
            coherence_integration._mock_olsson_class,
            'return_value',
            mock_olsson_instance
        ):
            variant = coherence_integration._init_coherence_variant("olsson")

            original_question = "What is the capital of France?"
            variations = [
                "Which city serves as France's capital?",
            ]

            score = coherence_integration.score_question_variations(
                variant, original_question, variations
            )

            # Should have called predict
            mock_olsson_instance.predict.assert_called_once()

            # Should return a float score
            assert isinstance(score, float)
            assert score == 0.68

    def test_score_normalization_range(self, coherence_integration):
        """Test that scores are in [0, 1] range after normalization."""
        # Mock variant that returns a score
        mock_variant = MagicMock()
        mock_variant.predict.return_value = np.array([0.95])

        original_question = "What is the capital of France?"
        variations = ["Which city is France's capital?"]

        score = coherence_integration.score_question_variations(
            mock_variant, original_question, variations
        )

        # Score should be in [0, 1] range
        assert 0.0 <= score <= 1.0

    def test_handles_edge_case_all_identical(self, coherence_integration):
        """Test handling of edge case where all variations are identical."""
        # Mock variant that returns maximum coherence for identical texts
        mock_variant = MagicMock()
        mock_variant.predict.return_value = np.array([1.0])

        original_question = "What is the capital of France?"
        # In practice, validation should prevent this, but test robustness
        variations = [original_question, original_question]

        score = coherence_integration.score_question_variations(
            mock_variant, original_question, variations
        )

        # Should handle gracefully and return high coherence
        assert isinstance(score, float)
        assert score == 1.0

    def test_handles_edge_case_diverse_variations(self, coherence_integration):
        """Test handling of highly diverse variations (low coherence)."""
        # Mock variant that returns low coherence for diverse texts
        mock_variant = MagicMock()
        mock_variant.predict.return_value = np.array([0.15])

        original_question = "What is the capital of France?"
        variations = [
            "How many planets are in the solar system?",  # Completely different
            "What is the weather like today?",
        ]

        score = coherence_integration.score_question_variations(
            mock_variant, original_question, variations
        )

        # Should handle gracefully and return low coherence
        assert isinstance(score, float)
        assert score == 0.15

    def test_error_handling_numerical_instability(self, coherence_integration):
        """Test error handling for numerical instability errors."""
        # Mock variant that raises a numerical error
        mock_variant = MagicMock()
        mock_variant.predict.side_effect = ValueError("Division by zero in coherence calculation")

        original_question = "What is the capital of France?"
        variations = ["Which city is France's capital?"]

        # Should handle error gracefully and return fallback score or raise with clear message
        with pytest.raises(ValueError, match="Division by zero"):
            coherence_integration.score_question_variations(
                mock_variant, original_question, variations
            )

    def test_error_handling_with_fallback(self, coherence_integration):
        """Test error handling with fallback behavior."""
        # Mock variant that raises an error
        mock_variant = MagicMock()
        mock_variant.predict.side_effect = RuntimeError("API error")

        original_question = "What is the capital of France?"
        variations = ["Which city is France's capital?"]

        # Should handle error and return fallback score
        score = coherence_integration.score_question_variations(
            mock_variant,
            original_question,
            variations,
            fallback_score=0.5
        )

        # Should return fallback score
        assert score == 0.5
