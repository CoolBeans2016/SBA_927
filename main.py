import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

import nltk
for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'averaged_perceptron_tagger_eng', 'maxent_ne_chunker', 'maxent_ne_chunker_tab', 'words']:
    nltk.download(pkg, quiet=True)

# ==================== SECTION 1: Text Preprocessing (noise removal, tokenization, stop words) ====================
# Function for Tokenization and Preprocessing
import re

def tokenize_and_preprocess(text):
    # Remove noise
    text = re.sub(r"[^A-Za-z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize the text
    tokens = word_tokenize(text)
      
    # Convert to lowercase
    tokens = [token.lower() for token in tokens]
    
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    filtered_tokens = [word.lower() for word in tokens if (word.isalpha() and word.lower() not in stop_words and word not in string.punctuation)
]
    return filtered_tokens

file_path = 'dataset.txt'
file_path = 'dataset.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    dataset = [line.strip() for line in file if line.strip()]

# Tokenization and Preprocessing of the entire dataset
processed_dataset = [tokenize_and_preprocess(text) for text in dataset]

# Display the results
for i, text_tokens in enumerate(processed_dataset):
    print(f"Original Text {i + 1}: {dataset[i]}")
    print(f"Processed Tokens {i + 1}: {text_tokens}")
    print("\n")

# ==================== SECTION 2: Stemming and Lemmatization ====================
# Import necessary libraries
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download NLTK resources if not already present
nltk.download('wordnet')

# Define a function for stemming and lemmatization
def stem_and_lemmatize(tokens):
    # Initialize stemming and lemmatization tools
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Applying stemming
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # Apply lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return stemmed_tokens, lemmatized_tokens

# Assume 'processed_dataset' contains the preprocessed text data

# Apply stemming and lemmatization to the entire dataset
stemmed_and_lemmatized_dataset = [stem_and_lemmatize(tokens) for tokens in processed_dataset]

# Display the results
for i, (stemmed_tokens, lemmatized_tokens) in enumerate(stemmed_and_lemmatized_dataset):
    print(f"Original Tokens {i + 1}: {dataset[i]}")
    print(f"Stemmed Tokens {i + 1}: {stemmed_tokens}")
    print(f"Lemmatized Tokens {i + 1}: {lemmatized_tokens}")
    print("\n")

# ==================== SECTION 3: Named Entity Recognition (NLTK) ====================
# Import necessary libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, ne_chunk

# Download required resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Function to tokenize text into sentences
def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

# Function to perform part-of-speech tagging on sentences
def part_or_speech_tagging(sentences):
    pos_tagged_sentences = [pos_tag(word_tokenize(sentence)) for sentence in sentences]
    return pos_tagged_sentences

# Function to extract named entities from POS-tagged sentences
def extract_named_entities(pos_tagged_sentences):
    named_entities = []
    for sentence in pos_tagged_sentences:
        tree = ne_chunk(sentence)
        for subtree in tree:
            if hasattr(subtree, 'label'): # Check if subtree is a named entity
                entity = ' '.join(word for word, tag in subtree.leaves())
                named_entities.append((entity, subtree.label()))
    return named_entities

# Apply Named Entity Recognition to the entire dataset
ner_results_dataset = [
    extract_named_entities(part_or_speech_tagging(nltk.sent_tokenize(text)))
    for text in dataset
]
# Display the results
for i, ner_results in enumerate(ner_results_dataset):
    print(f"Named Entities in Text {i + 1}: {ner_results}")
    print("\n")

# ==================== SECTION 4: Part-of-Speech Tagging ====================
# Import necessary libraries
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize

# Download NLTK resources if not already present
nltk.download('punkt')

# Assume 'processed_dataset' contains the preprocessed text data

# Define a function for POS tagging
def pos_tagging(text):
    # Tokenize and process the text
    sentences = sent_tokenize(text)
    pos_tags = []

# Process each sentence
    for sentence in sentences:
        words = word_tokenize(sentence)
        pos_tags.extend(pos_tag(words))

    return pos_tags 

# Apply POS tagging to the entire dataset
pos_tag_dataset = [pos_tagging(text) for text in dataset]

# Display the results
for i, tags in enumerate(pos_tag_dataset):
       print(f"POS Tags in Text{i+1}: {tags}")
       print("\n")

# ==================== SECTION 5: Sentiment Analysis (Hugging Face model and TextBlob) ====================
# NOTE: The five prompt-based sentiment analysis tests (zero-shot, role + strict format,
# few-shot, label + reason, Mixed + confidence) were run on Gemini Flash using the
# eight sentences in business_reviews.txt. The prompts, responses, and accuracy
# evaluation are documented in the report (Requirements 5 and 6).
from transformers import pipeline

# Load a pre-trained sentiment-analysis pipeline from Hugging Face
sentiment_pipeline = pipeline("sentiment-analysis")

# Hugging Face sentiment on business reviews
with open('business_reviews.txt', 'r', encoding='utf-8') as f:
    business = [line.strip() for line in f if line.strip()]

print("Hugging Face Sentiment on Business Reviews:\n")
for i, text in enumerate(business):
    r = sentiment_pipeline(text)[0]
    print(f"Text {i + 1}: {text}")
    print(f"Label: {r['label']}, Confidence Score: {round(r['score'], 4)}\n")

# Read the same movie dataset for Hugging Face analysis
file_path = 'dataset.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    dataset = [line.strip() for line in file if line.strip()]

# Run sentiment analysis on each line of the dataset
print("Hugging Face Sentiment Analysis Results:\n")
for i, text in enumerate(dataset):
    result = sentiment_pipeline(text)[0]
    print(f"Text {i + 1}: {text}")
    print(f"Label: {result['label']}, Confidence Score: {round(result['score'], 4)}")
    print()

from textblob import TextBlob

# Read the dataset
file_path = 'business_reviews.txt'
with open(file_path, 'r', encoding='utf-8') as file:
	dataset = [line.strip() for line in file if line.strip()]


# Function to interpret polarity score
def interpret_sentiment(polarity):
	if polarity > 0.1:
		return "Positive"
	elif polarity < -0.1:
		return "Negative"
	else:
		return "Neutral"


# Run sentiment analysis on each line of the dataset
print("NLTK/TextBlob Sentiment Analysis Results:\n")
for i, text in enumerate(dataset):
	blob = TextBlob(text)
	polarity = blob.sentiment.polarity
	subjectivity = blob.sentiment.subjectivity
	sentiment_label = interpret_sentiment(polarity)

	print(f"Text {i + 1}: {text}")
	print(f"Polarity: {round(polarity, 4)} | Subjectivity: {round(subjectivity, 4)} | Sentiment: {sentiment_label}")
	print()

# ==================== SECTION 6: Pre-trained Model Experiment (Hugging Face) ====================
import nltk
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import pipeline, AutoTokenizer

for pkg in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger',
            'averaged_perceptron_tagger_eng', 'maxent_ne_chunker',
            'maxent_ne_chunker_tab', 'words']:
    nltk.download(pkg, quiet=True)

# NLTK NER (rule/statistical chunker)
def nltk_entities(text):
    ents = []
    for sent in sent_tokenize(text):
        tree = ne_chunk(pos_tag(word_tokenize(sent)))
        for subtree in tree:
            if hasattr(subtree, 'label'):
                ents.append((" ".join(w for w, t in subtree.leaves()), subtree.label()))
    return ents

# Pre-trained transformer NER (BERT fine-tuned on CoNLL-2003)
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

def hf_entities(text):
    return [(e['word'], e['entity_group'], round(float(e['score']), 2)) for e in ner(text)]

with open('dataset.txt', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

print("NER COMPARISON: NLTK vs. Hugging Face (dslim/bert-base-NER)\n")
for i, text in enumerate(lines):
    print(f"Text {i + 1}: {text}")
    print(f"  NLTK:         {nltk_entities(text)}")
    print(f"  Hugging Face: {hf_entities(text)}\n")

# Pre-trained tokenizer vs. NLTK tokenizer on one sentence
sample = lines[1]
tok = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
print("TOKENIZATION COMPARISON")
print("Sentence:", sample)
print("NLTK word_tokenize:", word_tokenize(sample))
print("BERT subword tokens:", tok.tokenize(sample))

# ==================== SECTION 7: Prompt-Based Sentiment Analysis ====================
PROMPT_TEMPLATE = (
    "You are a business analyst. Classify the sentiment of the following business text "
    "as exactly one of: Positive, Negative, or Neutral. Respond with only the label.\n\n"
    'Text: "{text}"\nSentiment:'
)


# Replace these with the actual sentiment labels matching each line of business_reviews.txt in order
ground_truth = [
    "Positive",
    "Negative",
    "Neutral",
    "Positive",
    "Negative",
    "Negative",
    "Neutral",
    "Positive",
]

from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "google/flan-t5-base"
prompt_tokenizer = T5Tokenizer.from_pretrained(model_name)
prompt_model = T5ForConditionalGeneration.from_pretrained(model_name)



rows = []
print("PROMPT-BASED SENTIMENT ANALYSIS (flan-t5-base)\n")
for text, truth in zip(business, ground_truth):
    prompt = PROMPT_TEMPLATE.format(text=text)
    inputs = prompt_tokenizer(prompt, return_tensors="pt")
    output_ids = prompt_model.generate(**inputs, max_new_tokens=5)
    raw_out = prompt_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    prompt_label = raw_out.strip().capitalize()

    hf_label = sentiment_pipeline(text)[0]["label"].capitalize()
    tb_label = interpret_sentiment(TextBlob(text).sentiment.polarity)

    print(f"Text: {text}")
    print(f"  Truth: {truth} | TextBlob: {tb_label} | HF: {hf_label} | Prompt: {prompt_label}\n")
    rows.append((truth, tb_label, hf_label, prompt_label))
    
    
    
for name, col in [("TextBlob", 1), ("Hugging Face", 2), ("Prompt-based", 3)]:
    correct = sum(r[0] == r[col] for r in rows)
    print(f"  {name}: {correct}/{len(rows)}")
   



