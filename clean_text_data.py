import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Download NLTK resources if not already present
nltk.download('punkt')
nltk.download('stopwords')

# Function for Tokenization and Preprocessing
def tokenize_and_preprocess(text):
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

    # Read the external file

file_path ='dataset.txt'
with open(file_path, 'r', encoding='utf-8') as file: dataset = file.read().splitlines()

# Tokenization and Preprocessing of the entire dataset
processed_dataset = [tokenize_and_preprocess(text) for text in dataset]

# Display the results
for i, text_tokens in enumerate(processed_dataset):
    print(f"Original Text {i + 1}: {dataset[i]}")
    print(f"Processed Tokens {i + 1}: {text_tokens}")
    print("\n")

# Import necessary libraries
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download NLTK resources if not already present
nltk.download('wordnet')

# Define a function for stemming and lemmatization
def stem_and_lemmatize(tokens):
    # Initialize stemming an dlemmatization tools
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
for i, pos_tag in enumerate(pos_tag_dataset):
    print(f"POS Tags in Text{i+1}: {pos_tag}")
    print("\n")






   



