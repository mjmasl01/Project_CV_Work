# ds562team9-FinalProject

## Bitcoin Market Trends & Sentiment Analysis

**Team Members:**  
Aashrey Jain  
Matthew Maslow  
Ishanay Sharma  

**Course:** DS 598 – Engineering Big Data

---

## Project Overview

This project explores the volatility of the cryptocurrency market—specifically Bitcoin—by building a real-time analytics pipeline on Microsoft Azure. The system integrates live market data and social sentiment to forecast price fluctuations and inform trading decisions.

---

## Problem Statement

- Bitcoin price movements are highly influenced by public sentiment and real-time macroeconomic news.
- Traditional financial analysis techniques lack the ability to detect or respond to sudden sentiment-driven shifts.
- There is a gap in existing models regarding the integration of live sentiment data into market prediction systems.

---

## Objective

To develop an early warning system powered by:

- Twitter and financial news sentiment analysis  
- Live Bitcoin price and trading volume metrics  
- Machine learning predictions for short-term returns  
- Interactive dashboards for real-time insights

---

## System Architecture

This project utilizes Microsoft Azure services for:

- **Data Ingestion:** APIs for Twitter, GNews, and yFinance  
- **Processing:** Azure Functions, Azure Event Hubs  
- **Storage:** Azure Data Lake using the Medallion architecture (Bronze, Silver, Gold)  
- **Modeling:** HistGradientBoostingRegressor with Bayesian optimization  
- **Visualization:** Power BI dashboards for analysis and decision support

---

## Exploratory Data Analysis (EDA)

- Analyzed historical tweet and news volume related to Bitcoin (2013–2021)  
- Identified peaks in sentiment and their alignment with major Bitcoin price events  
- Assessed how trading volume and media coverage evolved alongside price trends

---

## Machine Learning Model

**Model Used:** Bayesian-optimized HistGradientBoostingRegressor

- **Input Features:**  
  - Rolling and daily sentiment from Twitter and news  
  - Bitcoin return, volatility, RSI, and SMA  
- **Target:** Next-day return (continuous value)  
- **Trading Decision Logic:**  
  - Forecast ≥ +0.1% → Buy  
  - Forecast ≤ −0.1% → Sell  
  - Otherwise → Hold

---

## Power BI Dashboard

**Dashboard Includes:**

1. **Sentiment vs. Volatility:**  
   - 3-day and 7-day rolling comparison of sentiment and market volatility

2. **Bitcoin Price vs. RSI:**  
   - Tracks price trends alongside the 14-day Relative Strength Index

3. **RSI vs. Price by Year:**  
   - Scatter plot by year with gauge chart for average RSI

4. **Yearly Sentiment vs. Price Averages:**  
   - Sentiment and price comparison across years to identify market cycles

---

## Key Findings

- Short-term sentiment shifts are often followed by changes in market volatility.
- RSI proved to be a strong indicator of momentum and reversals.
- Positive sentiment peaks aligned with major price surges in 2017 and 2021.

---

## Limitations

- Results are based on historical data only (not deployed in real-time production).
- Basic sentiment scoring may misclassify sarcasm or complex linguistic nuance.
- Data imbalance from high-activity periods or dominant sources may skew trends.

---

## Data Sources

- [Bitcoin Tweets (16M tweets)](https://www.kaggle.com/datasets/gauravduttakiit/bitcoin-tweets-16m-tweets-with-sentiment-tagged)  
- [Cryptocurrency News (2013–2018)](https://www.kaggle.com/datasets/kashnitsky/news-about-major-cryptocurrencies-20132018-40k)

---

## View the Dashboard

Access the live Power BI report here:  
**[Bitcoin Sentiment Dashboard](https://app.powerbi.com/view?r=eyJrIjoiY2JiZWM1MGItNzY1Ny00NWJiLWI4Y2EtZjE2ZWExNzFiZWY0IiwidCI6ImQ1N2QzMmNjLWMxMjEtNDg4Zi1iMDdiLWRmZTcwNTY4MGM3MSIsImMiOjN9)**


