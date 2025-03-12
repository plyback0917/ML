import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import requests
from bs4 import BeautifulSoup

# Simulated "scraping" function (replace with real scraping logic)
def scrape_sp500_metrics():
    # Placeholder values based on March 11, 2025 discussion
    metrics = {
        'Trailing_PE': 28.0,          # From prior discussion
        'Shiller_CAPE': 36.5,         # Approx current value
        'Dividend_Yield': 1.3,        # Low end of historical range
        'VIX': 16.5,                  # Mid-range complacency
        'Earnings_Yield': 3.57,       # 1 / Trailing_PE
        'Price_to_Sales': 2.95,       # Near all-time highs
        'Put_Call_Ratio': 0.9,        # Neutral sentiment
        'Breadth_200DMA': 62.5        # % above 200-day MA
    }
    
    return metrics

# Synthetic historical data for training (simplified example)
def generate_historical_data():
    np.random.seed(42)
    n_samples = 120  # 10 years of monthly data
    data = {
        'Trailing_PE': np.random.normal(20, 5, n_samples),
        'Shiller_CAPE': np.random.normal(25, 5, n_samples),
        'Dividend_Yield': np.random.normal(2, 0.5, n_samples),
        'VIX': np.random.normal(20, 5, n_samples),
        'Earnings_Yield': np.random.normal(5, 1, n_samples),
        'Price_to_Sales': np.random.normal(2, 0.5, n_samples),
        'Put_Call_Ratio': np.random.normal(1, 0.2, n_samples),
        'Breadth_200DMA': np.random.normal(70, 10, n_samples),
        'Growth': np.random.choice([0, 1], n_samples, p=[0.4, 0.6])  # 0 = reduction, 1 = growth
    }
    # Adjust Growth to reflect valuation extremes (high CAPE -> more likely reduction)
    for i in range(n_samples):
        if data['Shiller_CAPE'][i] > 30 or data['Trailing_PE'][i] > 25:
            data['Growth'][i] = np.random.choice([0, 1], p=[0.7, 0.3])
        elif data['Shiller_CAPE'][i] < 15:
            data['Growth'][i] = np.random.choice([0, 1], p=[0.2, 0.8])
    return pd.DataFrame(data)

# Train model and predict probability
def predict_growth_probability(current_metrics, historical_data):
    # Features and target
    features = ['Trailing_PE', 'Shiller_CAPE', 'Dividend_Yield', 'VIX', 
                'Earnings_Yield', 'Price_to_Sales', 'Put_Call_Ratio', 'Breadth_200DMA']
    X_train = historical_data[features]
    y_train = historical_data['Growth']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train logistic regression
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Prepare current metrics for prediction
    current_values = np.array([current_metrics[f] for f in features]).reshape(1, -1)
    current_scaled = scaler.transform(current_values)
    
    # Predict probability
    prob = model.predict_proba(current_scaled)[0]  # [P(reduction), P(growth)]
    return prob[1], prob[0]  # Return P(growth), P(reduction)

# Main execution
if __name__ == "__main__":
    # Step 1: "Scrape" current S&P 500 metrics
    current_metrics = scrape_sp500_metrics()
    print("Scraped Current Metrics:", current_metrics)
    
    # Step 2: Generate synthetic historical data (replace with real data source)
    historical_data = generate_historical_data()
    
    # Step 3: Predict probabilities
    prob_growth, prob_reduction = predict_growth_probability(current_metrics, historical_data)
    
    # Step 4: Output results
    print(f"\nProbability of S&P 500 Growth Next Month: {prob_growth:.2%}")
    print(f"Probability of S&P 500 Reduction Next Month: {prob_reduction:.2%}")