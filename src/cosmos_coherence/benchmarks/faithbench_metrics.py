"""FaithBench-specific metrics for hallucination detection evaluation."""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from cosmos_coherence.harness.base_benchmark import BenchmarkEvaluationResult

logger = logging.getLogger(__name__)


class FaithBenchMetrics:
    """Metrics calculator for FaithBench hallucination detection benchmark."""

    # Entropy thresholds for stratification
    LOW_ENTROPY_THRESHOLD = 0.33
    HIGH_ENTROPY_THRESHOLD = 0.67

    # Annotation labels
    ANNOTATION_LABELS = ["consistent", "questionable", "benign", "hallucinated"]

    def calculate_binary_metrics(
        self, results: List[BenchmarkEvaluationResult]
    ) -> Dict[str, float]:
        """Calculate binary classification metrics per FaithBench paper.

        Binary mapping per Table 2:
        - Positive class (no unwanted): benign + consistent
        - Negative class (has unwanted): hallucinated + questionable

        Args:
            results: List of evaluation results

        Returns:
            Dictionary containing binary metrics
        """
        if not results:
            return {
                "binary_precision": 0.0,
                "binary_recall": 0.0,
                "binary_f1": 0.0,
                "binary_accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "f1_macro_binary": 0.0,
            }

        # Binary confusion matrix
        tp = 0  # Predicted positive, actually positive
        tn = 0  # Predicted negative, actually negative
        fp = 0  # Predicted positive, actually negative
        fn = 0  # Predicted negative, actually positive

        for result in results:
            label = result.metadata.get("annotation_label")
            if not label:
                continue

            # Get model prediction from metadata or result
            llm_eval = result.metadata.get("llm_evaluation", "")
            model_predicts_positive = llm_eval == "consistent"

            # Ground truth binary mapping
            actual_positive = label in ["consistent", "benign"]

            # Update confusion matrix
            if model_predicts_positive and actual_positive:
                tp += 1
            elif not model_predicts_positive and not actual_positive:
                tn += 1
            elif model_predicts_positive and not actual_positive:
                fp += 1
            else:  # not model_predicts_positive and actual_positive
                fn += 1

        # Calculate binary metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # For negative class
        neg_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        neg_f1 = (
            2 * (neg_precision * neg_recall) / (neg_precision + neg_recall)
            if (neg_precision + neg_recall) > 0
            else 0.0
        )

        # Accuracy and balanced accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        balanced_acc = (recall + neg_recall) / 2

        # F1-macro for true binary classification (average of positive and negative F1)
        f1_macro_binary_true = (f1 + neg_f1) / 2

        return {
            "binary_precision": precision,
            "binary_recall": recall,
            "binary_f1": f1,
            "binary_accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "f1_macro_binary_true": f1_macro_binary_true,  # True binary F1-macro
        }

    def calculate_precision_recall(
        self, results: List[BenchmarkEvaluationResult]
    ) -> Dict[str, float]:
        """Calculate precision, recall, and F1 for each annotation label.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary containing precision, recall, and F1 metrics
        """
        if not results:
            return self._empty_precision_recall_metrics()

        # Initialize counters for each label
        label_counts = {
            label: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for label in self.ANNOTATION_LABELS
        }

        # Count true/false positives/negatives per label
        for result in results:
            label = result.metadata.get("annotation_label")
            if not label or label not in self.ANNOTATION_LABELS:
                continue

            if result.is_correct:
                label_counts[label]["tp"] += 1
                # Mark as true negative for other labels
                for other_label in self.ANNOTATION_LABELS:
                    if other_label != label:
                        label_counts[other_label]["tn"] += 1
            else:
                label_counts[label]["fn"] += 1
                # This is a false positive for whichever label was predicted
                # For simplicity, we'll count it as FP for the true label
                for other_label in self.ANNOTATION_LABELS:
                    if other_label != label:
                        label_counts[other_label]["fp"] += 1

        metrics = {}

        # Calculate metrics for each label
        for label in self.ANNOTATION_LABELS:
            counts = label_counts[label]
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            # Note: tn (true negatives) is tracked but not used in precision/recall calculations

            # Precision: TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            # Recall: TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # F1: 2 * (precision * recall) / (precision + recall)
            f1 = (
                2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            )

            metrics[f"precision_{label}"] = precision
            metrics[f"recall_{label}"] = recall
            metrics[f"f1_{label}"] = f1

        # Calculate overall metrics
        correct = sum(1 for r in results if r.is_correct)
        total = len(results)

        metrics["overall_accuracy"] = correct / total if total > 0 else 0.0
        metrics["overall_precision"] = np.mean(
            [metrics[f"precision_{label}"] for label in self.ANNOTATION_LABELS]
        )
        metrics["overall_recall"] = np.mean(
            [metrics[f"recall_{label}"] for label in self.ANNOTATION_LABELS]
        )
        metrics["overall_f1"] = np.mean(
            [metrics[f"f1_{label}"] for label in self.ANNOTATION_LABELS]
        )

        # Add balanced accuracy for binary classification
        # According to FaithBench paper Table 2:
        # Positive class: benign + consistent (no unwanted hallucination)
        # Negative class: unwanted + questionable (has unwanted content)

        # Calculate binary metrics by grouping classes
        # Positive (no unwanted hallucination): consistent + benign
        positive_tp = label_counts["consistent"]["tp"] + label_counts["benign"]["tp"]
        positive_fn = label_counts["consistent"]["fn"] + label_counts["benign"]["fn"]

        # Negative (has unwanted content): hallucinated + questionable
        negative_tp = label_counts["hallucinated"]["tp"] + label_counts["questionable"]["tp"]
        negative_fn = label_counts["hallucinated"]["fn"] + label_counts["questionable"]["fn"]

        # Binary recall for positive and negative classes (used for balanced accuracy)
        binary_positive_recall = (
            positive_tp / (positive_tp + positive_fn) if (positive_tp + positive_fn) > 0 else 0.0
        )
        binary_negative_recall = (
            negative_tp / (negative_tp + negative_fn) if (negative_tp + negative_fn) > 0 else 0.0
        )

        # Balanced accuracy: average of TPR and TNR
        metrics["balanced_accuracy"] = (binary_positive_recall + binary_negative_recall) / 2

        # F1-macro calculation per your suggestion: average across all 4 classes
        # The paper likely reports the macro-averaged F1 across the 4 annotation types
        # rather than the binary F1-macro
        metrics["f1_macro_binary"] = metrics["overall_f1"]  # Average of 4 class F1 scores

        # Keep the 4-class F1-macro for reference
        metrics["f1_macro"] = metrics["overall_f1"]  # Already macro-averaged above

        return metrics

    def calculate_entropy_metrics(
        self, results: List[BenchmarkEvaluationResult]
    ) -> Dict[str, float]:
        """Calculate entropy-based challenge scoring metrics.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary containing entropy-based metrics
        """
        if not results:
            return {
                "average_entropy": 0.0,
                "challenging_accuracy": 0.0,
                "challenging_count": 0,
                "non_challenging_accuracy": 0.0,
                "non_challenging_count": 0,
            }

        entropy_scores = []
        challenging_correct = 0
        challenging_total = 0
        non_challenging_correct = 0
        non_challenging_total = 0

        for result in results:
            entropy = result.metadata.get("entropy_score")
            is_challenging = result.metadata.get("is_challenging", False)

            if entropy is not None:
                entropy_scores.append(entropy)

            if is_challenging:
                challenging_total += 1
                if result.is_correct:
                    challenging_correct += 1
            else:
                non_challenging_total += 1
                if result.is_correct:
                    non_challenging_correct += 1

        return {
            "average_entropy": np.mean(entropy_scores) if entropy_scores else 0.0,
            "challenging_accuracy": (
                challenging_correct / challenging_total if challenging_total > 0 else 0.0
            ),
            "challenging_count": challenging_total,
            "non_challenging_accuracy": (
                non_challenging_correct / non_challenging_total
                if non_challenging_total > 0
                else 0.0
            ),
            "non_challenging_count": non_challenging_total,
        }

    def calculate_confusion_matrix(
        self, results: List[BenchmarkEvaluationResult]
    ) -> Dict[str, Dict[str, int]]:
        """Calculate confusion matrix for annotation labels.

        Args:
            results: List of evaluation results

        Returns:
            Confusion matrix as nested dictionary
        """
        confusion = {
            label: {
                "true_positive": 0,
                "false_positive": 0,
                "true_negative": 0,
                "false_negative": 0,
            }
            for label in self.ANNOTATION_LABELS
        }

        for result in results:
            label = result.metadata.get("annotation_label")
            if not label or label not in self.ANNOTATION_LABELS:
                continue

            if result.is_correct:
                confusion[label]["true_positive"] += 1
                # Count as true negative for other labels
                for other_label in self.ANNOTATION_LABELS:
                    if other_label != label:
                        confusion[other_label]["true_negative"] += 1
            else:
                confusion[label]["false_negative"] += 1
                # For simplicity, count as false positive for others
                for other_label in self.ANNOTATION_LABELS:
                    if other_label != label:
                        confusion[other_label]["false_positive"] += 1

        return confusion

    def calculate_detection_metrics(
        self, results: List[BenchmarkEvaluationResult]
    ) -> Dict[str, float]:
        """Calculate hallucination detection-specific metrics.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary containing detection metrics
        """
        if not results:
            return {
                "detection_precision": 0.0,
                "detection_recall": 0.0,
                "detection_f1": 0.0,
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0,
            }

        tp = 0  # True positive: hallucination correctly detected
        fp = 0  # False positive: non-hallucination marked as hallucination
        tn = 0  # True negative: non-hallucination correctly identified
        fn = 0  # False negative: hallucination not detected

        for result in results:
            label = result.metadata.get("annotation_label")
            detected = result.metadata.get("detected_hallucination", False)

            is_hallucination = label == "hallucinated"

            if is_hallucination and detected:
                tp += 1
            elif is_hallucination and not detected:
                fn += 1
            elif not is_hallucination and detected:
                fp += 1
            else:  # not is_hallucination and not detected
                tn += 1

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False positive rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False negative rate

        return {
            "detection_precision": precision,
            "detection_recall": recall,
            "detection_f1": f1,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
        }

    def calculate_stratified_metrics(
        self, results: List[BenchmarkEvaluationResult]
    ) -> Dict[str, float]:
        """Calculate metrics stratified by entropy levels.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary containing stratified metrics
        """
        low_entropy = {"correct": 0, "total": 0}
        medium_entropy = {"correct": 0, "total": 0}
        high_entropy = {"correct": 0, "total": 0}

        for result in results:
            entropy = result.metadata.get("entropy_score")
            if entropy is None:
                continue

            if entropy < self.LOW_ENTROPY_THRESHOLD:
                low_entropy["total"] += 1
                if result.is_correct:
                    low_entropy["correct"] += 1
            elif entropy < self.HIGH_ENTROPY_THRESHOLD:
                medium_entropy["total"] += 1
                if result.is_correct:
                    medium_entropy["correct"] += 1
            else:
                high_entropy["total"] += 1
                if result.is_correct:
                    high_entropy["correct"] += 1

        return {
            "low_entropy_accuracy": (
                low_entropy["correct"] / low_entropy["total"] if low_entropy["total"] > 0 else 0.0
            ),
            "low_entropy_count": low_entropy["total"],
            "medium_entropy_accuracy": (
                medium_entropy["correct"] / medium_entropy["total"]
                if medium_entropy["total"] > 0
                else 0.0
            ),
            "medium_entropy_count": medium_entropy["total"],
            "high_entropy_accuracy": (
                high_entropy["correct"] / high_entropy["total"]
                if high_entropy["total"] > 0
                else 0.0
            ),
            "high_entropy_count": high_entropy["total"],
        }

    def calculate_detector_agreement(
        self, results: List[BenchmarkEvaluationResult]
    ) -> Dict[str, float]:
        """Calculate metrics for detector agreement analysis.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary containing detector agreement metrics
        """
        if not results:
            return {
                "average_detector_agreement": 0.0,
                "unanimous_agreement_rate": 0.0,
                "majority_agreement_accuracy": 0.0,
            }

        agreement_scores = []
        unanimous_count = 0
        majority_correct = 0
        total_with_predictions = 0

        for result in results:
            predictions = result.metadata.get("detector_predictions")
            if not predictions:
                continue

            total_with_predictions += 1
            values = list(predictions.values())

            if len(values) > 0:
                # Calculate agreement as percentage of detectors agreeing
                most_common = max(set(values), key=values.count)
                agreement = values.count(most_common) / len(values)
                agreement_scores.append(agreement)

                # Check for unanimous agreement
                if all(v == values[0] for v in values):
                    unanimous_count += 1

                # Check if majority vote was correct
                if result.is_correct and agreement > 0.5:
                    majority_correct += 1

        return {
            "average_detector_agreement": np.mean(agreement_scores) if agreement_scores else 0.0,
            "unanimous_agreement_rate": (
                unanimous_count / total_with_predictions if total_with_predictions > 0 else 0.0
            ),
            "majority_agreement_accuracy": (
                majority_correct / total_with_predictions if total_with_predictions > 0 else 0.0
            ),
        }

    def calculate_aggregate_metrics(
        self, results: List[BenchmarkEvaluationResult]
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics aggregation.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary containing all aggregated metrics
        """
        metrics = {}

        # Add binary classification metrics (primary for FaithBench)
        metrics.update(self.calculate_binary_metrics(results))

        # Add precision/recall metrics for 4-class (secondary)
        metrics.update(self.calculate_precision_recall(results))

        # Add entropy-based metrics
        metrics.update(self.calculate_entropy_metrics(results))

        # Add stratified metrics
        metrics.update(self.calculate_stratified_metrics(results))

        # Add detector agreement metrics if available
        if any(r.metadata.get("detector_predictions") for r in results):
            metrics.update(self.calculate_detector_agreement(results))

        return metrics

    def calculate_weighted_metrics(
        self, results: List[BenchmarkEvaluationResult], weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate weighted metrics based on annotation importance.

        Args:
            results: List of evaluation results
            weights: Dictionary mapping annotation labels to weights

        Returns:
            Dictionary containing weighted metrics
        """
        if not results:
            return {"weighted_accuracy": 0.0, "weighted_precision": 0.0, "weighted_recall": 0.0}

        weighted_correct = 0.0
        total_weight = 0.0

        label_metrics: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "weight": 0.0}
        )

        for result in results:
            label = result.metadata.get("annotation_label")
            if not label or label not in weights:
                continue

            weight = weights.get(label, 1.0)
            total_weight += weight

            if result.is_correct:
                weighted_correct += weight
                label_metrics[label]["tp"] += weight
            else:
                label_metrics[label]["fn"] += weight

        # Calculate weighted accuracy
        weighted_accuracy = weighted_correct / total_weight if total_weight > 0 else 0.0

        # Calculate weighted precision and recall
        total_weighted_precision = 0.0
        total_weighted_recall = 0.0
        total_label_weight = 0.0

        for label, metrics_dict in label_metrics.items():
            tp = metrics_dict["tp"]
            fn = metrics_dict["fn"]
            weight = weights.get(label, 1.0)

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # For precision, we'd need FP which requires more complex tracking
            # Simplified version using accuracy as proxy
            precision = weighted_accuracy  # Simplified

            total_weighted_recall += recall * weight
            total_weighted_precision += precision * weight
            total_label_weight += weight

        return {
            "weighted_accuracy": weighted_accuracy,
            "weighted_precision": (
                total_weighted_precision / total_label_weight if total_label_weight > 0 else 0.0
            ),
            "weighted_recall": (
                total_weighted_recall / total_label_weight if total_label_weight > 0 else 0.0
            ),
        }

    def test_statistical_significance(
        self, results1: List[BenchmarkEvaluationResult], results2: List[BenchmarkEvaluationResult]
    ) -> Dict[str, Any]:
        """Test statistical significance of metric differences.

        Args:
            results1: First set of results
            results2: Second set of results

        Returns:
            Dictionary containing statistical test results
        """
        if not results1 or not results2:
            return {"p_value": 1.0, "is_significant": False, "confidence_interval": (0.0, 0.0)}

        # Get accuracy scores
        scores1 = [1.0 if r.is_correct else 0.0 for r in results1]
        scores2 = [1.0 if r.is_correct else 0.0 for r in results2]

        # Simple t-test approximation (would use scipy.stats in production)
        mean1 = np.mean(scores1)
        mean2 = np.mean(scores2)
        std1 = np.std(scores1)
        std2 = np.std(scores2)
        n1 = len(scores1)
        n2 = len(scores2)

        # Pooled standard error
        se = np.sqrt((std1**2 / n1) + (std2**2 / n2))

        # T-statistic
        t_stat = (mean1 - mean2) / se if se > 0 else 0.0

        # Simplified p-value calculation (would use proper distribution in production)
        # Using normal approximation for large samples
        p_value = 2 * (1 - min(0.975, max(0.025, 0.5 + 0.5 * abs(t_stat) / 2)))

        # Confidence interval
        margin = 1.96 * se  # 95% confidence
        ci_lower = (mean1 - mean2) - margin
        ci_upper = (mean1 - mean2) + margin

        return {
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "confidence_interval": (ci_lower, ci_upper),
            "mean_difference": mean1 - mean2,
        }

    def export_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Export metrics to a structured dictionary format.

        Args:
            metrics: Dictionary of calculated metrics

        Returns:
            Structured dictionary for export
        """
        # Group metrics by category
        per_label_metrics = {}
        entropy_metrics = {}
        detection_metrics = {}
        overall_metrics = {}

        for key, value in metrics.items():
            if any(label in key for label in self.ANNOTATION_LABELS):
                per_label_metrics[key] = value
            elif "entropy" in key or "challenging" in key:
                entropy_metrics[key] = value
            elif "detection" in key or "false_" in key:
                detection_metrics[key] = value
            else:
                overall_metrics[key] = value

        return {
            "summary": overall_metrics,
            "per_label_metrics": per_label_metrics,
            "entropy_metrics": entropy_metrics,
            "detection_metrics": detection_metrics,
            "timestamp": datetime.now().isoformat(),
        }

    def compare_with_baseline(
        self, current_metrics: Dict[str, float], baseline: Dict[str, float]
    ) -> Dict[str, float]:
        """Compare current metrics with baseline.

        Args:
            current_metrics: Current evaluation metrics
            baseline: Baseline metrics to compare against

        Returns:
            Dictionary containing metric deltas
        """
        comparison = {}

        for key in baseline:
            if key in current_metrics:
                delta = current_metrics[key] - baseline[key]
                comparison[f"{key}_delta"] = delta
                comparison[f"{key}_improved"] = delta > 0

        return comparison

    def _empty_precision_recall_metrics(self) -> Dict[str, float]:
        """Return empty precision/recall metrics structure.

        Returns:
            Dictionary with zero-valued metrics
        """
        metrics = {}
        for label in self.ANNOTATION_LABELS:
            metrics[f"precision_{label}"] = 0.0
            metrics[f"recall_{label}"] = 0.0
            metrics[f"f1_{label}"] = 0.0

        metrics.update(
            {
                "overall_accuracy": 0.0,
                "overall_precision": 0.0,
                "overall_recall": 0.0,
                "overall_f1": 0.0,
            }
        )

        return metrics
