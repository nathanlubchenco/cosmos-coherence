"""SimpleQA benchmark with coherence-guided hallucination prevention.

This module extends SimpleQABenchmark to implement question-variation coherence
for hallucination detection and prevention through strategic abstention.

Key innovation: Instead of measuring coherence across answer samples, this
approach measures coherence across question variations to detect ambiguous or
ill-posed questions that are likely to trigger hallucinations.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cosmos_coherence.benchmarks.implementations.simpleqa_benchmark import SimpleQABenchmark
from cosmos_coherence.benchmarks.implementations.simpleqa_grader import SimpleQAGrader
from cosmos_coherence.benchmarks.implementations.question_variation_generator import (
    QuestionVariationGenerator,
)
from cosmos_coherence.benchmarks.implementations.coherence_integration import (
    CoherenceIntegration,
)
from cosmos_coherence.benchmarks.implementations.answer_strategy import (
    AnswerStrategy,
)
from cosmos_coherence.benchmarks.models.datasets import SimpleQAItem
from cosmos_coherence.llm.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class SimpleQACoherenceBenchmark(SimpleQABenchmark):
    """SimpleQA benchmark with coherence-guided abstention.

    Extends SimpleQABenchmark to add question-variation coherence analysis
    for hallucination prevention. The core hypothesis: well-formed questions
    maintain high coherence across paraphrasing, while ambiguous or ill-posed
    questions show low coherence, indicating higher hallucination risk.

    The evaluation pipeline:
        1. Load SimpleQA question
        2. Generate question variations
        3. Calculate coherence score
        4. Determine answer strategy (YES/NO polarity or threshold-based abstention)
        5. Generate answer (if applicable)
        6. Grade answer
        7. Return result with metadata

    Example:
        >>> from cosmos_coherence.llm.openai_client import OpenAIClient
        >>> from cosmos_coherence.llm.config import OpenAIConfig
        >>>
        >>> config = OpenAIConfig(api_key="your-key", default_model="gpt-4o-mini")
        >>> client = OpenAIClient(config)
        >>>
        >>> benchmark = SimpleQACoherenceBenchmark(
        ...     client=client,
        ...     coherence_variant="shogenji",
        ...     coherence_threshold=0.5,
        ...     num_variations=5
        ... )
        >>>
        >>> # Evaluate dataset
        >>> results = await benchmark.evaluate_dataset(sample_size=100)
    """

    def __init__(
        self,
        client: OpenAIClient,
        coherence_variant: str = "shogenji",
        coherence_threshold: float = 0.5,
        num_variations: int = 5,
        temperature_range: Optional[List[float]] = None,
        abstention_response: str = "I don't know",
        model: str = "gpt-4o-mini",
        grader_model: str = "gpt-4o-mini",
        high_coherence_answer_yes: bool = True,
        verbose: bool = False,
        **kwargs
    ):
        """Initialize SimpleQA benchmark with coherence-guided abstention.

        Args:
            client: OpenAI client for API calls
            coherence_variant: Coherence measure to use ("shogenji", "fitelson", "olsson")
            coherence_threshold: Threshold for abstention decision (0.0-1.0, default: 0.5)
            num_variations: Number of question variations to generate (default: 5)
            temperature_range: Temperatures for variation generation (default: [0.7, 1.0, 1.3])
            abstention_response: Text to return when abstaining (default: "I don't know")
            model: Model for answer generation and variations (default: gpt-4o-mini)
            grader_model: Model for AI grading (default: gpt-4o-mini)
            high_coherence_answer_yes: For YES/NO questions, whether high coherence → YES
                (default: True, set False to flip polarity mapping)
            verbose: Enable verbose logging for debugging
            **kwargs: Additional arguments for parent SimpleQABenchmark
        """
        super().__init__(client=client, use_ai_grading=True, **kwargs)

        # Coherence configuration
        self.coherence_variant_name = coherence_variant
        self.coherence_threshold = coherence_threshold
        self.num_variations = num_variations
        self.temperature_range = temperature_range or [0.7, 1.0, 1.3]
        self.abstention_response = abstention_response
        self.model = model
        self.grader_model = grader_model
        self.high_coherence_answer_yes = high_coherence_answer_yes
        self.verbose = verbose

        # Initialize components
        self.variation_generator = QuestionVariationGenerator(
            client=client,
            model=model,
            temperature_range=self.temperature_range,
            num_variations=num_variations,
            verbose=verbose,
        )

        self.coherence_integration = CoherenceIntegration(
            model=model,
            # api_key will be extracted from environment by coherence variants
        )

        # Initialize coherence variant
        self.coherence_variant = self.coherence_integration._init_coherence_variant(
            coherence_variant
        )

        self.answer_strategy = AnswerStrategy(
            coherence_threshold=coherence_threshold,
            abstention_response=abstention_response,
            high_coherence_answer_yes=high_coherence_answer_yes,
            verbose=verbose,
        )

        # Initialize grader (lazy initialization inherited from parent)
        if not self._grader:
            self._grader = SimpleQAGrader(self.client, grader_model=grader_model)

    async def evaluate_with_coherence(self, item: SimpleQAItem) -> Dict:
        """Evaluate question with coherence-guided abstention.

        Implements the 8-step pipeline:
            1. Load SimpleQA question
            2. Generate question variations
            3. Generate answer sketches (original + variations)
            4. Calculate coherence score on answer sketches
            5. Determine answer strategy (YES/NO or threshold-based)
            6. Generate final answer (if applicable)
            7. Grade answer
            8. Return result with metadata

        Args:
            item: SimpleQA dataset item with question and ground truth

        Returns:
            Dictionary with evaluation results:
                - question: Original question
                - ground_truth: Expected answer
                - question_variations: Generated paraphrases
                - coherence_score: Coherence score on answer sketches (0.0-1.0)
                - threshold_met: Whether score >= threshold
                - question_type: "yes_no" or "factual"
                - final_answer: Answer after strategy applied
                - grade: AI grade (CORRECT/INCORRECT/NOT_ATTEMPTED)
                - is_correct: Boolean correctness
                - metadata: Additional tracking information including:
                    - original_answer_sketch: Answer used for coherence
                    - variation_answer_sketches: Variation answers for coherence
        """
        try:
            # Step 1: Load SimpleQA question (already provided as item)
            question = item.question
            ground_truth = item.best_answer

            if self.verbose:
                logger.info(f"\nEvaluating question: {question}")

            # Step 2: Generate question variations
            question_variations = await self.variation_generator.generate_variations(question)

            if self.verbose:
                logger.info(f"Generated {len(question_variations)} variations")

            # Step 3: Generate answer sketches (original + variations)
            # This is the theoretically sound approach: answers have truth values,
            # so probabilistic coherence measures are semantically appropriate.
            answer_sketch_prompt = "Provide a brief 1-2 sentence answer to this question: {question}"

            # Generate answer sketch for original question
            original_answer_response = await self.client.generate_response(
                prompt=answer_sketch_prompt.format(question=question),
                model=self.model,
                temperature=0.7,  # Moderate temperature for answer sketch
                max_tokens=100,
            )
            original_answer_sketch = original_answer_response.content.strip()

            # Generate answer sketches for variations
            variation_answer_sketches = []
            for variation in question_variations:
                var_response = await self.client.generate_response(
                    prompt=answer_sketch_prompt.format(question=variation),
                    model=self.model,
                    temperature=0.7,
                    max_tokens=100,
                )
                variation_answer_sketches.append(var_response.content.strip())

            if self.verbose:
                logger.info(f"Generated answer sketches - Original: {original_answer_sketch[:80]}...")
                logger.info(f"Generated {len(variation_answer_sketches)} variation answer sketches")

            # Step 4: Calculate coherence score on answer sketches
            coherence_score = self.coherence_integration.score_answer_sketches(
                coherence_variant=self.coherence_variant,
                original_answer=original_answer_sketch,
                variation_answers=variation_answer_sketches,
                fallback_score=0.5,  # Fallback on error
                verbose=self.verbose,
            )

            if self.verbose:
                logger.info(f"Answer sketch coherence score: {coherence_score:.4f}")

            # Step 5: Determine answer strategy
            question_type = "yes_no" if self.answer_strategy._is_yes_no_question(question) else "factual"
            threshold_met = coherence_score >= self.coherence_threshold

            # Step 6: Generate final answer (if needed)
            if question_type == "yes_no":
                # YES/NO questions: answer determined by coherence, no model call needed
                model_answer = None
            else:
                # Other questions: generate model answer
                response = await self.client.generate_response(
                    prompt=question,
                    model=self.model,
                    temperature=0.3,  # Low temperature for factual accuracy
                    max_tokens=50,
                )
                model_answer = response.content.strip()

                if self.verbose:
                    logger.info(f"Model answer: {model_answer}")

            # Apply answer strategy (handles YES/NO and threshold logic)
            final_answer = self.answer_strategy.apply_strategy(
                question=question,
                coherence_score=coherence_score,
                model_answer=model_answer,
            )

            if self.verbose:
                logger.info(f"Final answer after strategy: {final_answer}")

            # Step 7: Grade answer
            grade, grading_metadata = await self._grader.grade_response(
                question=question,
                expert_answer=ground_truth,
                submission=final_answer,
            )

            is_correct = grade == "CORRECT"

            if self.verbose:
                logger.info(f"Grade: {grade}, Correct: {is_correct}")

            # Step 8: Return result with metadata
            return {
                "question": question,
                "ground_truth": ground_truth,
                "question_variations": question_variations,
                "coherence_score": coherence_score,
                "threshold_met": threshold_met,
                "question_type": question_type,
                "final_answer": final_answer,
                "grade": grade,
                "is_correct": is_correct,
                "metadata": {
                    "num_variations": len(question_variations),
                    "coherence_variant": self.coherence_variant_name,
                    "coherence_threshold": self.coherence_threshold,
                    "grading_metadata": grading_metadata,
                    "model_answer": model_answer,  # Final model answer (before strategy)
                    "original_answer_sketch": original_answer_sketch,  # Answer sketch used for coherence
                    "variation_answer_sketches": variation_answer_sketches,  # Variation answers for coherence
                },
            }

        except Exception as e:
            logger.error(f"Error evaluating question: {question}, Error: {e}")
            # Return error result
            return {
                "question": question,
                "ground_truth": ground_truth,
                "question_variations": [],
                "coherence_score": 0.0,
                "threshold_met": False,
                "question_type": "unknown",
                "final_answer": f"ERROR: {str(e)}",
                "grade": "INCORRECT",
                "is_correct": False,
                "metadata": {
                    "error": str(e),
                },
            }

    async def evaluate_dataset(
        self,
        sample_size: Optional[int] = None,
        checkpoint_interval: int = 50,
        checkpoint_path: Optional[Path] = None,
        resume_from: Optional[Path] = None,
        workers: int = 1,
    ) -> List[Dict]:
        """Evaluate full SimpleQA dataset with coherence-guided abstention.

        Args:
            sample_size: Number of questions to evaluate (default: all 4,326)
            checkpoint_interval: Save checkpoint every N questions (default: 50, 0=disabled)
            checkpoint_path: Path for checkpoint file (default: auto-generated)
            resume_from: Resume from existing checkpoint file
            workers: Number of parallel workers (default: 1, use 1 for sequential)

        Returns:
            List of evaluation results (one per question)
        """
        # Load dataset
        dataset = await self.load_dataset()

        # Limit to sample size if specified
        if sample_size:
            dataset = dataset[:sample_size]

        logger.info(f"Evaluating {len(dataset)} questions with coherence-guided abstention")

        # Load checkpoint if resuming
        start_idx = 0
        results = []

        if resume_from and resume_from.exists():
            logger.info(f"Resuming from checkpoint: {resume_from}")
            results, start_idx = self.load_checkpoint(resume_from)
            start_idx += 1  # Continue from next question
            logger.info(f"Resuming from question {start_idx + 1}")

        # Generate default checkpoint path if not provided
        if checkpoint_path is None and checkpoint_interval > 0:
            checkpoint_path = Path(f"simpleqa_coherence_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        # Evaluate questions
        for idx, item in enumerate(dataset[start_idx:], start=start_idx):
            if isinstance(item, SimpleQAItem):
                try:
                    result = await self.evaluate_with_coherence(item)
                    results.append(result)

                    # Save checkpoint periodically
                    if checkpoint_interval > 0 and (idx + 1) % checkpoint_interval == 0:
                        if checkpoint_path:
                            self.save_checkpoint(
                                results=results,
                                checkpoint_path=checkpoint_path,
                                last_index=idx,
                                config={
                                    "model": self.model,
                                    "coherence_variant": self.coherence_variant_name,
                                    "coherence_threshold": self.coherence_threshold,
                                    "num_variations": self.num_variations,
                                    "sample_size": len(dataset),
                                }
                            )
                            logger.info(f"Checkpoint saved at question {idx + 1}")

                except Exception as e:
                    logger.error(f"Error on question {idx + 1}: {e}")
                    # Add error result and continue
                    results.append({
                        "question": item.question,
                        "ground_truth": item.best_answer,
                        "coherence_score": 0.0,
                        "final_answer": f"ERROR: {str(e)}",
                        "is_correct": False,
                    })

        # Clean up checkpoint on successful completion
        if checkpoint_interval > 0 and checkpoint_path and checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Checkpoint file removed after successful completion")

        return results

    def save_checkpoint(
        self,
        results: List[Dict],
        checkpoint_path: Path,
        last_index: int,
        config: Dict,
    ) -> None:
        """Save evaluation checkpoint to file.

        Args:
            results: Partial evaluation results
            checkpoint_path: Path to save checkpoint
            last_index: Index of last evaluated question
            config: Configuration parameters
        """
        checkpoint_data = {
            "config": config,
            "progress": {
                "questions_completed": len(results),
                "last_index": last_index,
                "timestamp": datetime.now().isoformat(),
            },
            "partial_results": results,
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    def load_checkpoint(self, checkpoint_path: Path) -> Tuple[List[Dict], int]:
        """Load checkpoint from file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Tuple of (results, last_index)
        """
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)

            results = checkpoint_data.get("partial_results", [])
            last_index = checkpoint_data.get("progress", {}).get("last_index", -1)

            return results, last_index

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return [], -1
