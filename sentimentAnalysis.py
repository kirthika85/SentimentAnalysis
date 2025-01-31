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
    return stock_data

# Function to validate sentiment against stock performance
def validate_sentiment(sentiment_score, stock_performance):
    stock_change = (stock_performance['Close'].iloc[-1] - stock_performance['Close'].iloc[0]) / stock_performance['Close'].iloc[0]
    volume_change = (stock_performance['Volume'].iloc[-1] - stock_performance['Volume'].iloc[0]) / stock_performance['Volume'].iloc[0]
    
    if (sentiment_score > 0.2 and stock_change < -0.05) or (sentiment_score < -0.2 and stock_change > 0.05):
        stock_discrepancy = "Potential discrepancy detected between sentiment and stock price."
    else:
        stock_discrepancy = "Sentiment aligns with stock price."
        
    if (sentiment_score > 0.2 and volume_change < 0) or (sentiment_score < -0.2 and volume_change > 0):
        volume_discrepancy = "Potential discrepancy detected between sentiment and trading volume."
    else:
        volume_discrepancy = "Sentiment aligns with trading volume."
    
    return stock_discrepancy, volume_discrepancy

def get_earnings_data(ticker, api_key):
    api_url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={api_key}"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        earnings_data = response.json()
        return earnings_data
    except requests.exceptions.HTTPError as errh:
        return f"HTTP Error occurred: {errh}. Check your API key and endpoint."
    except requests.exceptions.ConnectionError as errc:
        return f"Error connecting: {errc}"
    except requests.exceptions.Timeout as errt:
        return f"Timeout Error occurred: {errt}"
    except requests.exceptions.RequestException as err:
        return f"Something went wrong: {err}"
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit app
st.title("Earnings Call Sentiment Analysis")
LOGO_URL="Tesla-Logo.png"
st.image(LOGO_URL, width=200)

# Display the logo and dashboard
st.markdown("<h2 style='color: green;'>Company Logo and Dashboard</h2>", unsafe_allow_html=True)

if st.button("View Company Dashboard"):
    # Predefined values
    url = "https://wallstreetwaves.com/tesla-tsla-q4-2024-earnings-call-highlights-and-insights/"
    ticker = "TSLA"
    api_key="2K9GC7BCAV2V7RX8"
    
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
        stock_discrepancy, volume_discrepancy = validate_sentiment(overall_sentiment['compound'], stock_performance)
        
        st.subheader("Sentiment Validation")
        st.write(f"Stock Price: {stock_discrepancy}")
        st.write(f"Trading Volume: {volume_discrepancy}")
        
        # Visualize sentiment, stock price, and trading volume
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        ax1.hist([s['compound'] for s in sentiments], bins=20)
        ax1.set_xlabel("Sentiment Score")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Sentiment Scores")
        
        ax2.plot(stock_performance.index, stock_performance['Close'].values)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Stock Price")
        ax2.set_title(f"{ticker} Stock Price")
        
        ax3.plot(stock_performance.index, stock_performance['Volume'].values)
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Trading Volume")
        ax3.set_title(f"{ticker} Trading Volume")
        
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
        earnings_data = get_earnings_data(ticker, api_key)
        if isinstance(earnings_data, dict):
            st.write("Earnings Data:")
            if 'annualEarnings' in earnings_data:
                for year, data in earnings_data['annualEarnings'].items():
                    st.write(f"Year: {year}, EPS: {data.get('fiscalEps', 'N/A')}")
            else:
                st.write("Failed to retrieve specific earnings data.")
        else:
            st.write(earnings_data)
            
        # Display stock price
        stock_data = yf.Ticker(ticker).info
        stock_price = stock_data['currentPrice']
        st.write(f"Stock Price: {stock_price}")
        
        # Display sentiment analysis
        st.write(f"Overall Sentiment: {overall_sentiment['compound']:.2f}")
