
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Load stock data
def get_stock_data(ticker, period="6mo"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df["Return"] = df["Close"].pct_change()
    return df

# Scrape news headlines
def get_news_headlines(ticker):
    query = f"{ticker} stock"
    url = f"https://news.google.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    headlines = [h.text for h in soup.find_all('a') if h.text.strip() != ""]
    return headlines[:5]

# FinBERT sentiment analysis
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment(headlines, classifier):
    sentiments = classifier(headlines)
    scores = [1 if s['label'] == 'positive' else -1 if s['label'] == 'negative' else 0 for s in sentiments]
    return np.mean(scores)

# Recommendation logic
def make_recommendation(df, sentiment_score):
    short_ma = df['Close'].rolling(window=5).mean().iloc[-1]
    long_ma = df['Close'].rolling(window=20).mean().iloc[-1]
    current_price = df['Close'].iloc[-1]

    if short_ma > long_ma and sentiment_score > 0:
        return "BUY", f"Positive trend with strong sentiment. Price: {current_price:.2f}"
    elif short_ma < long_ma and sentiment_score < 0:
        return "SELL", f"Negative trend and weak sentiment. Price: {current_price:.2f}"
    else:
        return "HOLD", f"Unclear trend. Mixed indicators. Price: {current_price:.2f}"

# Streamlit UI
st.title("AI-Powered Stock Recommender")
st.markdown("Enter comma-separated tickers to get Buy/Sell/Hold recommendations.")

tickers = st.text_input("Stock Tickers", "AAPL, MSFT, TSLA")

if st.button("Analyze"):
    classifier = load_sentiment_pipeline()
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    
    for ticker in ticker_list:
        st.subheader(f"Analysis for {ticker}")
        try:
            df = get_stock_data(ticker)
            headlines = get_news_headlines(ticker)
            sentiment_score = analyze_sentiment(headlines, classifier)
            reco, explanation = make_recommendation(df, sentiment_score)

            st.markdown(f"**Recommendation:** `{reco}`")
            st.markdown(f"**Reason:** {explanation}")
            st.markdown("**Recent News Headlines:**")
            for h in headlines:
                st.write(f"- {h}")
        except Exception as e:
            st.error(f"Error analyzing {ticker}: {e}")
