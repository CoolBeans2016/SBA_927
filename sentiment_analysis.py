from textblob import TextBlob

# Read the dataset
file_path = 'dataset.txt'
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

