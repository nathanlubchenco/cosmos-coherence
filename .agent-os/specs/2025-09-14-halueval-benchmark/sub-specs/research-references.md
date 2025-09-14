# HaluEval Research References

## Paper Information

### Primary Paper
- **Title**: HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models
- **Authors**: Junyi Li, Xiaoxue Cheng, Xin Zhao, Jian-Yun Nie, Ji-Rong Wen
- **Conference**: EMNLP 2023
- **ArXiv**: https://arxiv.org/abs/2305.11747
- **ACL Anthology**: https://aclanthology.org/2023.emnlp-main.397/
- **PDF**: https://aclanthology.org/2023.emnlp-main.397.pdf

### Paper Versions
- v1: May 2023 (initial submission)
- v2: October 2023 (camera-ready version)

## Repository Resources

### Official GitHub Repository
- **URL**: https://github.com/RUCAIBox/HaluEval
- **License**: MIT
- **Stars**: 100+ (as of 2024)
- **Language**: Python

### Repository Structure
```
HaluEval/
├── data/                    # 35,000 hallucination samples
│   ├── qa_data.json        # 10,000 QA samples
│   ├── dialogue_data.json  # 10,000 dialogue samples
│   ├── summarization_data.json  # 10,000 summarization samples
│   └── general_data.json   # 5,000 general samples
├── evaluation/              # Evaluation scripts
│   ├── evaluate.py         # Main evaluation script
│   ├── qa/
│   │   └── qa_evaluation_instruction.txt
│   ├── dialogue/
│   │   └── dialogue_evaluation_instruction.txt
│   └── summarization/
│       └── summarization_evaluation_instruction.txt
├── generation/             # Hallucination generation scripts
└── analysis/              # Result analysis tools
```

## Key Implementation Details

### Evaluation Prompts

#### QA Task
```
I want you to act as an answer judge. Given a question and an answer,
your objective is to determine if the provided answer contains non-factual
or hallucinated information. You SHOULD give your judgement based on the
following hallucination types and the world knowledge.

You are trying to determine if the answer misunderstands the question
context and intention.

#Question#: {question}
#Answer#: {answer}
#Your Judgement#: Yes or No
```

#### Dialogue Task
```
I want you to act as a response judge. Given a dialogue history and a response,
your objective is to determine if the provided response contains non-factual
or hallucinated information.

Hallucination types to check:
1. Replacing a true entity with a highly similar entity
2. Replacing a true entity with a dissimilar entity
3. Replacing a true entity with a dissimilar entity of a different type

#Dialogue History#: {dialogue_history}
#Response#: {response}
#Your Judgement#: Yes or No
```

#### Summarization Task
```
I want you to act as a summary judge. Given a document and a summary,
your objective is to determine if the provided summary contains non-factual
or hallucinated information.

Check for:
1. Information not directly inferrable from the document
2. Incorrect information
3. Contradictions with the document

#Document#: {document}
#Summary#: {summary}
#Your Judgement#: Yes or No
```

## Dataset Statistics

### Sample Distribution
- **Total Samples**: 35,000
- **QA Tasks**: 10,000 samples (28.6%)
- **Dialogue Tasks**: 10,000 samples (28.6%)
- **Summarization Tasks**: 10,000 samples (28.6%)
- **General Tasks**: 5,000 samples (14.2%)

### Data Sources
- **QA**: HotpotQA dataset
- **Dialogue**: OpenDialKG dataset
- **Summarization**: CNN/Daily Mail dataset
- **General**: Alpaca instruction dataset

### Generation Method
- **Framework**: ChatGPT-based two-step process
- **Strategy 1**: One-pass generation
- **Strategy 2**: Conversational generation
- **Filtering**: Human annotation for quality control

## Evaluation Metrics

### Primary Metrics
- **Accuracy**: Overall correctness of hallucination detection
- **Precision**: Correct hallucination identifications / Total hallucination predictions
- **Recall**: Correct hallucination identifications / Total actual hallucinations
- **F1 Score**: Harmonic mean of precision and recall

### Task-Specific Performance (from paper)
- **QA Tasks**: ~65-70% accuracy for GPT-3.5
- **Dialogue Tasks**: ~60-65% accuracy for GPT-3.5
- **Summarization Tasks**: ~70-75% accuracy for GPT-3.5
- **General Tasks**: ~55-60% accuracy for GPT-3.5

## Model Configuration

### Recommended Settings (from repository)
```python
{
    "model": "gpt-3.5-turbo",  # or "text-davinci-003"
    "temperature": 0.0,         # Deterministic responses
    "max_tokens": 10,          # Only need "Yes" or "No"
    "top_p": 1.0,
    "frequency_penalty": 0,
    "presence_penalty": 0
}
```

### API Requirements
- OpenAI API access required
- Estimated tokens per evaluation: ~500-1000 (varies by task)
- Cost consideration: Use caching for development

## Implementation Notes

### Critical Requirements
1. **Exact Prompt Reproduction**: Must use exact prompts from instruction files
2. **Binary Response**: Strictly "Yes" or "No" responses only
3. **Random Selection**: Randomly choose between right/hallucinated answers
4. **Deterministic Evaluation**: Temperature=0 for consistency
5. **Task-Specific Handling**: Different prompts for each task type

### Common Pitfalls to Avoid
1. Don't modify the evaluation prompts
2. Don't add explanations to the prompts
3. Don't use different response formats
4. Don't skip the random selection step
5. Don't forget task-specific context (dialogue history, document, etc.)

## Related Work

### HaluEval 2.0 (2024)
- Extended to 5 domains: biomedicine, finance, science, education, open domain
- Available at: https://github.com/RUCAIBox/HaluEval-2.0

### Other Hallucination Benchmarks
- **TruthfulQA**: Focus on truthfulness in QA
- **FaithBench**: Faithfulness in summarization
- **FEVER**: Fact extraction and verification
- **SimpleQA**: Simple factual questions

## Citation

```bibtex
@inproceedings{li-etal-2023-halueval,
    title = "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models",
    author = "Li, Junyi and Cheng, Xiaoxue and Zhao, Xin and Nie, Jian-Yun and Wen, Ji-Rong",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.397",
    pages = "6449--6464",
}
```
