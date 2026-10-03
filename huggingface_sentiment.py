from transformers import pipeline

# Load a pre-trained sentiment-analysis pipeline from Hugging Face
sentiment_pipeline = pipeline("sentiment-analysis")

# Read the same dataset used in the NLTK/TextBlob analysis
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