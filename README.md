# SBA 927 - Business Text Analytics: Natural Language Processing Techniques

**Student:** Crystal Marie
**Date:** September 30, 2026

## Overview

For this assignment, I used two text files for different parts of the analysis. The movie-related `dataset.txt` file contains 12 non-empty text entries. I used it for NLTK preprocessing, stemming and lemmatization, named entity recognition (NER), and part-of-speech (POS) tagging. I also compared NLTK NER and tokenization with Hugging Face pre-trained models.



The `business_reviews.txt` file contains eight business reviews. I analyzed these reviews with TextBlob, a Hugging Face sentiment-analysis pipeline, and a prompt-based experiment using Google's `google/flan-t5-base` model. I compared each model's three-way sentiment label with reference labels that I assigned to the reviews.

## Methods and Workflow

### Text preprocessing

I loaded the movie text from `dataset.txt`, removed characters outside the selected letters, numbers, whitespace, and basic punctuation, and collapsed extra whitespace. I tokenized the text with NLTK, converted tokens to lowercase, removed punctuation and English stop words, and retained alphabetic tokens. This prepares the words for later linguistic analysis.

### Stemming and lemmatization

I applied NLTK's Porter stemmer and WordNet lemmatizer to the preprocessed tokens. Stemming reduces words to shorter stems, which are not always dictionary words. Lemmatization aims to return a dictionary form. The two approaches can therefore produce different results.

### NER and POS tagging

I used NLTK sentence and word tokenization, POS tagging, and `ne_chunk` to identify and categorize named entities in the movie text. I also applied NLTK POS tagging to the text. POS tags identify grammatical roles such as nouns, verbs, and adjectives; NER attempts to identify entities such as people and organizations.

### Pre-trained model experiment

I compared NLTK's NER output with the Hugging Face `dslim/bert-base-NER` model on the movie text. I also compared NLTK word tokenization with the model's BERT subword tokenizer on a sample sentence. NLTK generally represents whole words, while a BERT tokenizer may split a word into smaller pieces.

The POS tagging in this script uses NLTK's `pos_tag`; I did not run a separate Hugging Face POS model. The Hugging Face experiment in this script is the NER and tokenization comparison, as well as the sentiment analysis described below.

### Sentiment methods and reference labels

I evaluated the same eight business reviews with three methods:

1. **TextBlob:** I used its polarity score and classified a review as Positive when polarity was greater than 0.1, Negative when it was less than -0.1, and Neutral otherwise.
2. **Hugging Face:** I used the default `sentiment-analysis` pipeline and recorded the returned label and confidence score.
3. **Prompt-based FLAN-T5:** I used `google/flan-t5-base` to classify each review according to a prompt requesting exactly one of Positive, Negative, or Neutral.

My reference labels are human judgments for this small evaluation set. The fifth review contains both praise and criticism; I labeled it Negative because the setup and support complaints outweigh “works fine.”

| Review | Reference label | Basis for label |
|---|---|---|
| 1 | Positive | The app saves the team time. |
| 2 | Negative | The customer was charged twice and received no response. |
| 3 | Neutral | The review gives a factual update about a report. |
| 4 | Positive | The delivery was quicker than expected. |
| 5 | Negative | Setup and support complaints outweigh the mild praise. |
| 6 | Negative | The customer complains that an update breaks the product. |
| 7 | Neutral | The review gives factual subscription information. |
| 8 | Positive | The customer's issue was resolved and they recommend the service. |

## Prompt-Based Sentiment Analysis

The prompt template in `main.py` is:

> You are a business analyst. Classify the sentiment of the following business text as exactly one of: Positive, Negative, or Neutral. Respond with only the label.
>
> Text: "{text}"
> Sentiment:

The script inserts each business review in place of `{text}` and asks `google/flan-t5-base` to generate a response. In my run, the prompt-based method matched 7 of the 8 reference labels. A short, strict prompt makes the expected output easy to compare, but a response still needs to be checked against the review and reference labels.

## Sentiment Results and Evaluation

In my latest run of `main.py`, each of the eight reviews was evaluated once. The results were:

| Method | Correct reference labels | Accuracy |
|---|---:|---:|
| TextBlob | 5/8 | 62.5% |
| Hugging Face sentiment pipeline | 5/8 | 62.5% |
| Prompt-based `google/flan-t5-base` | 7/8 | 87.5% |

These percentages describe only this eight-review sample. They do not establish that one method will be more accurate on other business text. The reference labels are human-assigned, and different reasonable judgments—especially for mixed feedback—could change the scores.

TextBlob's classification depends on the polarity thresholds selected in my code. Its simple scoring is easy to run, but a polarity score may not reflect context such as sarcasm or a complaint phrased without strongly negative vocabulary.

The Hugging Face sentiment pipeline returns a label and confidence score. The default sentiment model is generally a positive/negative classifier rather than a three-class model. If it returns only Positive and Negative, it cannot directly predict either Neutral reference case. Its score is a model confidence value, not proof that the prediction is correct.

The prompt-based method performed best against my labels in this run. It can follow an instruction that explicitly includes Neutral, but its output can still be wrong or inconsistent. The model's result should be evaluated on labeled examples rather than accepted solely because it follows the requested format.

## Comparison and Limitations

| Technique | Strengths observed or supported by the method | Limitations |
|---|---|---|
| NLTK preprocessing, stemming, and lemmatization | Makes text-processing steps visible and produces normalized tokens for later analysis. | Removing stop words and punctuation can remove context; stems may not be real words, and lemmatization without grammatical context can be limited. |
| NLTK POS tagging and NER | Produces grammatical tags and named-entity candidates using a local NLP library. | Tags and entity boundaries can be imperfect, especially for fictional names, titles, or text unlike the model's training examples. |
| TextBlob | Simple to run and provides polarity and subjectivity values. | Thresholds are chosen by the programmer; a lexicon-based score can miss context, sarcasm, or mixed sentiment. |
| Hugging Face sentiment pipeline | Uses a pre-trained classifier and provides a confidence score with its prediction. | The default classifier may not have a Neutral class, and confidence does not guarantee correctness on this business-review sample. |
| Prompt-based FLAN-T5 | The instruction can specify the desired categories and response format. | Results depend on the prompt and model; generated labels need validation, and this small evaluation does not show general performance. |
| Hugging Face NER and BERT tokenization | Provides a pre-trained NER comparison and illustrates subword tokenization. | Entity recognition and token boundaries can still be incorrect on the movie text; the tokenizer's subwords are not equivalent to human-readable words. |

## Conclusion

On the eight business reviews in this run, TextBlob and the Hugging Face sentiment pipeline each matched 5 of 8 reference labels (62.5%). The prompt-based `google/flan-t5-base` experiment matched 7 of 8 (87.5%). This is evidence about the methods' results on this particular sample, not a general ranking of their accuracy. The small dataset, human-assigned reference labels, and mixed review limit the conclusions.

The experiments show that different NLP techniques serve different purposes. NLTK supports inspectable preprocessing and linguistic analysis; TextBlob gives a simple polarity measure; Hugging Face provides pre-trained classification and NER; and prompted FLAN-T5 follows a natural-language classification instruction. Each method has limitations, so I would check model outputs against labeled examples before relying on them for business decisions. Debugging the pipeline also reinforced that data must be tokenized and tagged in the correct order before NER can process it.

## Appendix

The complete, current Python implementation is in [`main.py`](./main.py). The input files are [`dataset.txt`](./dataset.txt) and [`business_reviews.txt`](./business_reviews.txt). screenshots are included in the report and the code and input files are in main.py, dataset.txt, and business_reviews.txt.
