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

def get_earnings_data(ticker, api_key):
    api_url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-estimates"
    headers = {
        'x-rapidapi-host': "seeking-alpha.p.rapidapi.com",
        'x-rapidapi-key': api_key
    }
    params = {
        "symbol": ticker,
        "data_type": "eps",
        "period_type": "quarterly"
    }
    
    response = requests.get(api_url, headers=headers, params=params)
    
    if response.status_code == 200:
        earnings_data = response.json()
        return earnings_data
    else:
        return "Failed to retrieve earnings data."

# Streamlit app
st.title("Earnings Call Sentiment Analysis")
LOGO_URL="Tesla-Logo.png"
st.image(LOGO_URL, width=200)


# Display the logo as a button
col1, col2 = st.columns([2, 1])
with col1:
    st.write("")
    st.markdown("<h2 style='color: green;'>Company Logo and Dashboard</h2>", unsafe_allow_html=True)
    
    if st.button("View Company Dashboard"):
        # Predefined values
        url = "https://wallstreetwaves.com/tesla-tsla-q4-2024-earnings-call-highlights-and-insights/"
        ticker = "TSLA"
        api_key="0b7ddfd1d5msh644b6045b584129p1455edjsn2c8bbec1be38"
        
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
            
            # Display dashboard
            st.subheader("Company Dashboard")
            
            # Display earnings
            earnings = "Insert earnings data here"
            st.write(f"Earnings: {earnings}")
            
            # Display stock price
            stock_data = yf.Ticker(ticker).info
            stock_price = stock_data['currentPrice']
            st.write(f"Stock Price: {stock_price}")
            
            # Display sentiment analysis
            st.write(f"Overall Sentiment: {overall_sentiment['compound']:.2f}")
    
with col2:
    # Sidebar content here
    st.write("")
    st.markdown("<h2 style='color: green;'>About</h2>", unsafe_allow_html=True)
    st.write("This app analyzes earnings call transcripts to provide sentiment analysis and validate it against stock performance.")
