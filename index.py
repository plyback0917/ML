import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Function to scrape current S&P 500 numerical properties
def scrape_sp500_metrics():
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Scrape S&P 500 P/E and Dividend Yield from Yahoo Finance
    url_yahoo = "https://finance.yahoo.com/quote/%5EGSPC/"
    response = requests.get(url_yahoo, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    pe_ratio = float(soup.find("td", {"data-test": "PE_RATIO-value"}).text.replace(',', '')) if soup.find("td", {"data-test": "PE_RATIO-value"}) else 28.0
    div_yield = float(soup.find("td", {"data-test": "DIVIDEND_AND_YIELD-value"}).text.split('(')[1].replace('%)', '')) if soup.find("td", {"data-test": "DIVIDEND_AND_YIELD-value"}) else 1.3

    # Scrape Shiller CAPE from Multpl
    url_multpl = "https://www.multpl.com/shiller-pe"
    response = requests.get(url_multpl, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    cape = float(soup.find(id="current_value").text) if soup.find(id="current_value") else 36.5

    # Scrape VIX from Yahoo Finance
    url_vix = "https://finance.yahoo.com/quote/%5EVIX/"
    response = requests.get(url_vix, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    vix = float(soup.find("fin-streamer", {"data-field": "regularMarketPrice"}).text) if soup.find("fin-streamer", {"data-field": "regularMarketPrice"}) else 16.0

    # Simulated values for metrics not easily scraped in real-time
    market_breadth = 62.0  # % above 200-day MA (simulated, real scraping requires paid API)
    earnings_yield = 1 / pe_ratio * 100  # Inverse of P/E
    ps_ratio = 2.95  # Price-to-Sales (simulated, from prior data)
    put_call = 0.9  # Put/Call ratio (simulated)

    return {
        "PE_Ratio": pe_ratio,
        "Dividend_Yield": div_yield,
        "Shiller_CAPE": cape,
        "VIX": vix,
        "Market_Breadth": market_breadth,
        "Earnings_Yield": earnings_yield,
        "Price_Sales": ps_ratio,
        "Put_Call": put_call
    }

# Simulated historical data (for training, replace with real data if available)
def generate_historical_data():
    np.random.seed(42)
    data = {
        "PE_Ratio": [25, 20, 30, 28, 22, 26, 29, 27, 23, 28],
        "Dividend_Yield":​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​