"""
Pretrained Models for NLP Tasks

Experiments:
- Perform sentiment classification with a pretrained classifier.
- Extract named entities using a pretrained token-classification model.
- Answer questions using a pretrained extractive question-answering model.
- Generate text using a pretrained causal language model.
- Compare task-specific outputs and confidence scores.
"""

from transformers import pipeline


CLASSIFICATION_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
NER_MODEL = "dslim/bert-base-NER"
QA_MODEL = "distilbert/distilbert-base-cased-distilled-squad"
GENERATION_MODEL = "openai-community/gpt2"


def run_text_classification() -> None:
    """Predict sentiment labels for input sentences."""
    classifier = pipeline(
        "text-classification",
        model=CLASSIFICATION_MODEL,
    )

    texts = [
        "The new language model performs remarkably well.",
        "The application is slow and frustrating to use.",
    ]

    print("\n--- Text Classification ---")
    for result in classifier(texts):
        print(
            f"Label: {result['label']}, "
            f"score: {result['score']:.4f}"
        )


def run_named_entity_recognition() -> None:
    """Identify named entities in text."""
    recognizer = pipeline(
        "token-classification",
        model=NER_MODEL,
        aggregation_strategy="simple",
    )

    text = (
        "OpenAI develops AI technologies, and researchers work "
        "in Berlin and Mannheim."
    )

    print("\n--- Named Entity Recognition ---")
    print("Text:", text)

    for entity in recognizer(text):
        print(
            f"Entity: {entity['word']!r}, "
            f"label: {entity['entity_group']}, "
            f"score: {entity['score']:.4f}"
        )


def run_question_answering() -> None:
    """Extract an answer from a supplied context."""
    qa = pipeline(
        "question-answering",
        model=QA_MODEL,
    )

    context = (
        "Mannheim is a city in southwestern Germany. "
        "It is located in the state of Baden-Württemberg. "
        "The city is known for its grid-like city center."
    )

    questions = [
        "In which country is Mannheim located?",
        "In which German state is Mannheim located?",
        "What is the city center known for?",
    ]

    print("\n--- Extractive Question Answering ---")
    for question in questions:
        result = qa(question=question, context=context)
        print(
            f"Question: {question}\n"
            f"Answer: {result['answer']}\n"
            f"Score: {result['score']:.4f}\n"
        )


def run_text_generation() -> None:
    """Generate a continuation from a text prompt."""
    generator = pipeline(
        "text-generation",
        model=GENERATION_MODEL,
    )

    prompts = [
        "Pretrained language models are useful because",
        "A reliable AI system should",
    ]

    print("\n--- Text Generation ---")
    for prompt in prompts:
        result = generator(
            prompt,
            max_new_tokens=40,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=generator.tokenizer.eos_token_id,
        )[0]

        print("Prompt:", prompt)
        print("Generated:", result["generated_text"])
        print()


def main() -> None:
    run_text_classification()
    run_named_entity_recognition()
    run_question_answering()
    run_text_generation()


if __name__ == "__main__":
    main()
