import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from transformers import pipeline

def get_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    texts = soup.find_all(text=True)
    visible_texts = filter(tag_visible, texts)
    return " ".join(t.strip() for t in visible_texts)

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity

def analyze_sentiment_transformers(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

if __name__ == "__main__":
    url = input("Enter URL to analyze sentiment: ")
    text = get_text_from_url(url)
    method = input("Choose method (textblob/transformers): ").strip().lower()

    if method == "textblob":
        polarity, subjectivity = analyze_sentiment_textblob(text)
        print(f"TextBlob - Polarity: {polarity}, Subjectivity: {subjectivity}")
    elif method == "transformers":
        label, score = analyze_sentiment_transformers(text)
        print(f"Transformers - Label: {label}, Score: {score}")
    else:
        print("Invalid method chosen. Please select either 'textblob' or 'transformers'.")
