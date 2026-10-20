# Advanced NLP Tasks and Document Understanding

## Learning Objectives

- Apply token classification and named entity recognition (NER) for structured information extraction.
- Understand long-document classification, chunking, and hierarchical prediction aggregation.
- Implement extractive question answering and evidence-grounding checks.
- Integrate multiple NLP models into a reusable document understanding pipeline.
- Evaluate extraction quality, answer correctness, and document-level predictions.

## Core Concepts

### 1. Token Classification and Named Entity Recognition

Token classification assigns a label to each token. In BIO tagging, `B-` identifies the beginning of an entity, `I-` identifies a continuation, and `O` indicates a token outside an entity.

For a sequence of tokens, the model predicts:

$$
\hat{y}_i = \arg\max_{c} P(y_i=c \mid x)
$$

Here, $$x$$ is the input sequence and $$c$$ is a possible token label.

Entity extraction combines token predictions into structured records, such as people, organizations, and locations.

### 2. Long-Document Classification

Transformer models have a finite input context. Long documents can be divided into overlapping chunks:

$$
C_i = (w_{i s}, \ldots, w_{i s+L-1})
$$

Here, $$L$$ is the chunk length and $$s$$ is the stride between consecutive chunks. Overlap helps preserve information near chunk boundaries.

Chunk-level predictions can be aggregated into a document-level decision. For majority voting:

$$
\hat{y}_{\text{doc}} =
\arg\max_c \sum_{i=1}^{n} \mathbf{1}(\hat{y}_i=c)
$$

This approach is simple, but it may ignore differences in chunk importance and prediction confidence.

### 3. Question Answering and Document Grounding

Extractive question answering predicts an answer span within a supplied context. A model estimates the start and end positions of the answer:

$$
\hat{s} = \arg\max_i P_{\text{start}}(i \mid q,x)
$$

$$
\hat{e} = \arg\max_j P_{\text{end}}(j \mid q,x)
$$

Here, $$q$$ is the question and $$x$$ is the context.

Grounding checks verify that an answer corresponds to source evidence. However, an answer appearing in a document does not by itself prove that the answer correctly addresses the question.

### 4. Information Extraction and Evaluation

Exact-match entity evaluation requires both the entity text and its type to match the reference annotation.

$$
\text{Precision} = \frac{TP}{TP+FP}
$$

$$
\text{Recall} = \frac{TP}{TP+FN}
$$

$$
F_1 = 2 \cdot \frac{\text{Precision}\cdot\text{Recall}}
{\text{Precision}+\text{Recall}}
$$

Here, $$TP$$ represents correct predictions, $$FP$$ represents incorrect extractions, and $$FN$$ represents missed entities.

For question answering, exact match and token-level F1 provide complementary measures of answer quality. Neither metric alone fully captures semantic correctness or evidence quality.

## Implementation and Experiments

| File                                                        | Experiment                                                                                                                                                  |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_token_classification_and_information_extraction.py`     | Inspect token-level predictions, BIO-style labels, entity aggregation, structured extraction, and exact-match entity precision, recall, and F1.             |
| `02_document_classification_and_hierarchical_processing.py` | Compare truncation, overlapping chunks, mean-confidence aggregation, and majority voting for long-document classification.                                  |
| `03_question_answering_and_document_grounding.py`           | Extract answer spans, inspect confidence scores and character offsets, validate source correspondence, and evaluate answers against references.             |
| `04_document_understanding_pipeline.py`                     | Integrate preprocessing, chunking, classification, entity extraction, and question answering into a reusable pipeline that produces structured JSON output. |

The experiments use pretrained Hugging Face Transformers models and illustrative document examples. Their results demonstrate implementation techniques rather than establishing production-level accuracy.

## Expected Outcome

After completing this section, you should be able to:

- Extract structured entities from unstructured text.
- Explain the limitations of truncation and design basic long-document processing strategies.
- Implement extractive question answering with source-evidence checks.
- Combine independent NLP components into a document processing pipeline.
- Evaluate information extraction and answer quality using appropriate metrics.
- Identify engineering improvements needed for reliable document intelligence, including source-offset preservation, representative evaluation data, and stronger evidence validation.
