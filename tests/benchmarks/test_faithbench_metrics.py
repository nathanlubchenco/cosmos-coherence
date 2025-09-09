"""Tests for FaithBench metrics implementation."""

from typing import List

import pytest
from cosmos_coherence.harness.base_benchmark import BenchmarkEvaluationResult


class TestFaithBenchMetrics:
    """Test suite for FaithBench-specific metrics."""

    @pytest.fixture
    def sample_results_mixed(self) -> List[BenchmarkEvaluationResult]:
        """Create mixed evaluation results for testing."""
        return [
            # Consistent - correctly identified
            BenchmarkEvaluationResult(
                is_correct=True,
                score=0.9,
                original_metric_score=0.85,
                metadata={
                    "annotation_label": "consistent",
                    "entropy_score": 0.1,
                    "is_challenging": False,
                },
            ),
            # Hallucinated - correctly identified
            BenchmarkEvaluationResult(
                is_correct=True,
                score=0.8,
                original_metric_score=0.2,
                metadata={
                    "annotation_label": "hallucinated",
                    "entropy_score": 0.8,
                    "is_challenging": True,
                },
            ),
            # Questionable - incorrectly identified
            BenchmarkEvaluationResult(
                is_correct=False,
                score=0.3,
                original_metric_score=0.4,
                metadata={
                    "annotation_label": "questionable",
                    "entropy_score": 0.6,
                    "is_challenging": False,
                },
            ),
            # Benign - correctly identified
            BenchmarkEvaluationResult(
                is_correct=True,
                score=0.7,
                original_metric_score=0.65,
                metadata={
                    "annotation_label": "benign",
                    "entropy_score": 0.3,
                    "is_challenging": False,
                },
            ),
            # Hallucinated - incorrectly identified (false negative)
            BenchmarkEvaluationResult(
                is_correct=False,
                score=0.1,
                original_metric_score=0.9,
                metadata={
                    "annotation_label": "hallucinated",
                    "entropy_score": 0.9,
                    "is_challenging": True,
                },
            ),
        ]

    @pytest.fixture
    def faithbench_metrics(self):
        """Create FaithBenchMetrics instance."""
        from cosmos_coherence.benchmarks.faithbench_metrics import FaithBenchMetrics

        return FaithBenchMetrics()

    def test_calculate_precision_recall_per_label(self, faithbench_metrics, sample_results_mixed):
        """Test precision and recall calculation for each annotation label."""
        metrics = faithbench_metrics.calculate_precision_recall(sample_results_mixed)

        # Check that we have metrics for all labels
        assert "precision_consistent" in metrics
        assert "recall_consistent" in metrics
        assert "f1_consistent" in metrics

        assert "precision_hallucinated" in metrics
        assert "recall_hallucinated" in metrics
        assert "f1_hallucinated" in metrics

        # Verify specific calculations
        # For hallucinated: 1 TP, 1 FN -> recall = 0.5
        assert metrics["recall_hallucinated"] == 0.5

    def test_calculate_overall_precision_recall(self, faithbench_metrics, sample_results_mixed):
        """Test overall precision and recall across all labels."""
        metrics = faithbench_metrics.calculate_precision_recall(sample_results_mixed)

        assert "overall_precision" in metrics
        assert "overall_recall" in metrics
        assert "overall_f1" in metrics

        # Overall: 3 correct out of 5 = 0.6 accuracy
        assert metrics["overall_accuracy"] == 0.6

    def test_calculate_entropy_based_metrics(self, faithbench_metrics, sample_results_mixed):
        """Test entropy-based challenge scoring."""
        metrics = faithbench_metrics.calculate_entropy_metrics(sample_results_mixed)

        assert "average_entropy" in metrics
        assert "challenging_accuracy" in metrics
        assert "challenging_count" in metrics
        assert "non_challenging_accuracy" in metrics

        # 2 challenging samples, 1 correct -> 0.5 accuracy
        assert metrics["challenging_accuracy"] == 0.5
        assert metrics["challenging_count"] == 2

    def test_calculate_confusion_matrix(self, faithbench_metrics, sample_results_mixed):
        """Test confusion matrix generation for annotation labels."""
        confusion = faithbench_metrics.calculate_confusion_matrix(sample_results_mixed)

        assert isinstance(confusion, dict)
        assert "consistent" in confusion
        assert "hallucinated" in confusion
        assert "questionable" in confusion
        assert "benign" in confusion

        # Each label should have TP, FP, TN, FN
        for label in confusion:
            assert "true_positive" in confusion[label]
            assert "false_positive" in confusion[label]
            assert "true_negative" in confusion[label]
            assert "false_negative" in confusion[label]

    def test_calculate_detection_metrics(self, faithbench_metrics):
        """Test hallucination detection-specific metrics."""
        results = [
            # True hallucination, detected
            BenchmarkEvaluationResult(
                is_correct=True,
                score=0.2,
                original_metric_score=0.1,
                metadata={
                    "annotation_label": "hallucinated",
                    "detected_hallucination": True,
                },
            ),
            # Not hallucination, correctly not detected
            BenchmarkEvaluationResult(
                is_correct=True,
                score=0.9,
                original_metric_score=0.95,
                metadata={
                    "annotation_label": "consistent",
                    "detected_hallucination": False,
                },
            ),
            # True hallucination, not detected (false negative)
            BenchmarkEvaluationResult(
                is_correct=False,
                score=0.8,
                original_metric_score=0.85,
                metadata={
                    "annotation_label": "hallucinated",
                    "detected_hallucination": False,
                },
            ),
        ]

        metrics = faithbench_metrics.calculate_detection_metrics(results)

        assert "detection_precision" in metrics
        assert "detection_recall" in metrics
        assert "detection_f1" in metrics
        assert "false_positive_rate" in metrics
        assert "false_negative_rate" in metrics

        # 1 TP, 1 FN for hallucination detection
        assert metrics["detection_recall"] == 0.5

    def test_calculate_stratified_metrics(self, faithbench_metrics):
        """Test metrics stratified by entropy levels."""
        results = []
        # Low entropy samples (0.0 - 0.33)
        for i in range(3):
            results.append(
                BenchmarkEvaluationResult(
                    is_correct=True,
                    score=0.9,
                    original_metric_score=0.9,
                    metadata={"entropy_score": 0.1, "annotation_label": "consistent"},
                )
            )

        # Medium entropy samples (0.33 - 0.67)
        for i in range(3):
            results.append(
                BenchmarkEvaluationResult(
                    is_correct=i < 2,  # 2 correct, 1 incorrect
                    score=0.5,
                    original_metric_score=0.5,
                    metadata={"entropy_score": 0.5, "annotation_label": "questionable"},
                )
            )

        # High entropy samples (0.67 - 1.0)
        for i in range(3):
            results.append(
                BenchmarkEvaluationResult(
                    is_correct=i < 1,  # 1 correct, 2 incorrect
                    score=0.3,
                    original_metric_score=0.3,
                    metadata={"entropy_score": 0.8, "annotation_label": "hallucinated"},
                )
            )

        metrics = faithbench_metrics.calculate_stratified_metrics(results)

        assert "low_entropy_accuracy" in metrics
        assert "medium_entropy_accuracy" in metrics
        assert "high_entropy_accuracy" in metrics

        assert metrics["low_entropy_accuracy"] == 1.0  # All correct
        assert metrics["medium_entropy_accuracy"] == pytest.approx(2 / 3)
        assert metrics["high_entropy_accuracy"] == pytest.approx(1 / 3)

    def test_calculate_aggregate_metrics(self, faithbench_metrics, sample_results_mixed):
        """Test comprehensive metric aggregation."""
        metrics = faithbench_metrics.calculate_aggregate_metrics(sample_results_mixed)

        # Should include all metric types
        assert "overall_accuracy" in metrics
        assert "precision_hallucinated" in metrics
        assert "recall_hallucinated" in metrics
        assert "average_entropy" in metrics
        assert "challenging_accuracy" in metrics

        # Should have metrics for all annotation labels
        for label in ["consistent", "questionable", "benign", "hallucinated"]:
            assert f"precision_{label}" in metrics
            assert f"recall_{label}" in metrics
            assert f"f1_{label}" in metrics

    def test_empty_results_handling(self, faithbench_metrics):
        """Test metrics calculation with empty results."""
        metrics = faithbench_metrics.calculate_aggregate_metrics([])

        assert metrics["overall_accuracy"] == 0.0
        assert metrics["average_entropy"] == 0.0
        assert metrics["challenging_count"] == 0

    def test_missing_metadata_handling(self, faithbench_metrics):
        """Test handling of results with missing metadata."""
        results = [
            BenchmarkEvaluationResult(
                is_correct=True,
                score=0.5,
                original_metric_score=0.5,
                metadata={},  # Missing annotation_label and entropy_score
            )
        ]

        # Should not crash
        metrics = faithbench_metrics.calculate_aggregate_metrics(results)
        assert "overall_accuracy" in metrics

    def test_detector_agreement_metrics(self, faithbench_metrics):
        """Test metrics for detector agreement analysis."""
        results = [
            BenchmarkEvaluationResult(
                is_correct=True,
                score=0.8,
                original_metric_score=0.8,
                metadata={
                    "annotation_label": "hallucinated",
                    "detector_predictions": {
                        "gpt-4-turbo": 1,
                        "gpt-4o": 1,
                        "claude-3": 0,
                    },
                    "entropy_score": 0.5,
                },
            ),
            BenchmarkEvaluationResult(
                is_correct=True,
                score=0.9,
                original_metric_score=0.9,
                metadata={
                    "annotation_label": "consistent",
                    "detector_predictions": {
                        "gpt-4-turbo": 0,
                        "gpt-4o": 0,
                        "claude-3": 0,
                    },
                    "entropy_score": 0.0,
                },
            ),
        ]

        metrics = faithbench_metrics.calculate_detector_agreement(results)

        assert "average_detector_agreement" in metrics
        assert "unanimous_agreement_rate" in metrics
        assert "majority_agreement_accuracy" in metrics

    def test_export_metrics_to_dict(self, faithbench_metrics, sample_results_mixed):
        """Test exporting metrics to dictionary format."""
        metrics = faithbench_metrics.calculate_aggregate_metrics(sample_results_mixed)
        exported = faithbench_metrics.export_metrics(metrics)

        assert isinstance(exported, dict)
        assert "summary" in exported
        assert "per_label_metrics" in exported
        assert "entropy_metrics" in exported
        assert "timestamp" in exported

    def test_compare_with_baseline(self, faithbench_metrics, sample_results_mixed):
        """Test comparison with baseline metrics."""
        baseline = {
            "overall_accuracy": 0.7,
            "precision_hallucinated": 0.8,
            "recall_hallucinated": 0.75,
            "challenging_accuracy": 0.6,
        }

        current_metrics = faithbench_metrics.calculate_aggregate_metrics(sample_results_mixed)
        comparison = faithbench_metrics.compare_with_baseline(current_metrics, baseline)

        assert "overall_accuracy_delta" in comparison
        assert "precision_hallucinated_delta" in comparison
        assert "recall_hallucinated_delta" in comparison
        assert "challenging_accuracy_delta" in comparison

        # Check that deltas are calculated correctly
        assert comparison["overall_accuracy_delta"] == pytest.approx(
            current_metrics["overall_accuracy"] - baseline["overall_accuracy"]
        )

    def test_calculate_weighted_metrics(self, faithbench_metrics):
        """Test weighted metrics based on annotation importance."""
        results = [
            # Hallucinated - high weight
            BenchmarkEvaluationResult(
                is_correct=True,
                score=0.2,
                original_metric_score=0.2,
                metadata={"annotation_label": "hallucinated"},
            ),
            # Benign - low weight
            BenchmarkEvaluationResult(
                is_correct=False,
                score=0.7,
                original_metric_score=0.7,
                metadata={"annotation_label": "benign"},
            ),
        ]

        # Weight hallucinated errors more heavily
        weights = {"hallucinated": 2.0, "benign": 0.5, "consistent": 1.0, "questionable": 1.5}

        metrics = faithbench_metrics.calculate_weighted_metrics(results, weights)

        assert "weighted_accuracy" in metrics
        assert "weighted_precision" in metrics
        assert "weighted_recall" in metrics

    def test_statistical_significance(self, faithbench_metrics):
        """Test statistical significance of metric differences."""
        results1 = [
            BenchmarkEvaluationResult(
                is_correct=True, score=0.9, original_metric_score=0.9, metadata={}
            )
            for _ in range(10)
        ]

        results2 = [
            BenchmarkEvaluationResult(
                is_correct=i < 6, score=0.6, original_metric_score=0.6, metadata={}
            )
            for i in range(10)
        ]

        significance = faithbench_metrics.test_statistical_significance(results1, results2)

        assert "p_value" in significance
        assert "is_significant" in significance
        assert "confidence_interval" in significance
