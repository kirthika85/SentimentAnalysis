import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Function to scrape transcript (unchanged)
def scrape_transcript(url):
    # ... (your existing scrape_transcript function)

# Function to perform sentiment analysis (unchanged)
def analyze_sentiment(text):
    # ... (your existing analyze_sentiment function)

# Function to get stock performance
def get_stock_performance(ticker, days_before=30, days_after=30):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_before + days_after)
    stock_data = yf.Ticker(ticker).history(start=start_date, end=end_date)
    return stock_data['Close']

# Function to validate sentiment against stock performance
def validate_sentiment(sentiment_score, stock_performance):
    stock_change = (stock_performance.iloc[-1] - stock_performance.iloc[0]) / stock_performance.iloc[0]
    
    if (sentiment_score > 0.2 and stock_change < -0.05) or (sentiment_score < -0.2 and stock_change > 0.05):
        return "Potential discrepancy detected between sentiment and stock performance."
    else:
        return "Sentiment aligns with stock performance."

# Streamlit app
st.title("Earnings Call Sentiment Analysis and Validation")

# User input
url = st.text_input("Enter the URL of the earnings call transcript:")
ticker = st.text_input("Enter the stock ticker symbol:")

if url and ticker:
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
            
            # Get stock performance
            stock_performance = get_stock_performance(ticker)
            
            # Validate sentiment
            validation_result = validate_sentiment(overall_sentiment['compound'], stock_performance)
            
            st.subheader("Sentiment Validation")
            st.write(validation_result)
            
            # Visualize sentiment and stock performance
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            ax1.hist([s['compound'] for s in sentiments], bins=20)
            ax1.set_xlabel("Sentiment Score")
            ax1.set_ylabel("Frequency")
            ax1.set_title("Distribution of Sentiment Scores")
            
            ax2.plot(stock_performance.index, stock_performance.values)
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Stock Price")
            ax2.set_title(f"{ticker} Stock Performance")
            
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
