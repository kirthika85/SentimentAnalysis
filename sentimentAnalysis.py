import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt


# Download NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Function to scrape transcript
def scrape_transcript(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Look for common transcript container classes
    transcript_containers = soup.find_all(['div', 'section'], class_=['transcript-text', 'article-text', 'transcript-container'])
    
    if not transcript_containers:
        # If no specific container found, try to get all paragraphs
        transcript_containers = soup.find_all('p')
    
    transcript = ' '.join([container.get_text(strip=True) for container in transcript_containers])
    
    # Remove any video player text or irrelevant content
    irrelevant_phrases = ['Video Player', 'Loading Video', 'Transcript', 'Q&A Session']
    for phrase in irrelevant_phrases:
        transcript = transcript.replace(phrase, '')
    
    return transcript.strip()

# Function to perform sentiment analysis
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentences = nltk.sent_tokenize(text)
    sentiments = [sia.polarity_scores(sentence) for sentence in sentences]
    return sentiments

# Streamlit app
st.title("Earnings Call Sentiment Analysis")

# User input
url = st.text_input("Enter the URL of the earnings call transcript:")

if url:
    try:
        # Scrape transcript
        transcript = scrape_transcript(url)
        
        if not transcript:
            st.error("Unable to extract transcript from the provided URL. Please check the URL and try again.")
        else:
            # Display a sample of the scraped text
            st.subheader("Sample of Scraped Text")
            st.write(transcript[:500] + "...")  # Display first 500 characters
            
            # Perform sentiment analysis
            sentiments = analyze_sentiment(transcript)
            
            # Calculate overall sentiment
            overall_sentiment = pd.DataFrame(sentiments).mean()
            
            # Display results
            st.subheader("Overall Sentiment")
            st.write(f"Positive: {overall_sentiment['pos']:.2f}")
            st.write(f"Neutral: {overall_sentiment['neu']:.2f}")
            st.write(f"Negative: {overall_sentiment['neg']:.2f}")
            st.write(f"Compound: {overall_sentiment['compound']:.2f}")
            
            # Visualize sentiment distribution
            fig, ax = plt.subplots()
            ax.hist([s['compound'] for s in sentiments], bins=20)
            ax.set_xlabel("Sentiment Score")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Sentiment Scores")
            st.pyplot(fig)
            
            # Display most positive and negative sentences
            df_sentiments = pd.DataFrame(sentiments)
            df_sentiments['sentence'] = nltk.sent_tokenize(transcript)
            
            st.subheader("Most Positive Sentences")
            st.table(df_sentiments.nlargest(5, 'compound')[['sentence', 'compound']])
            
            st.subheader("Most Negative Sentences")
            st.table(df_sentiments.nsmallest(5, 'compound')[['sentence', 'compound']])
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
