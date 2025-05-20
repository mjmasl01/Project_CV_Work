# CEO Pay and Performance Dashboard - Documentation

## Overview
This Streamlit dashboard visualizes the relationship between CEO compensation and company performance metrics. It allows users to explore patterns, identify outliers, and analyze trends through interactive visualizations.

## Features
- Time series analysis of CEO pay and performance metrics
- Scatter plots with regression lines showing pay-performance relationships
- Correlation heatmaps between various financial and executive metrics
- Comparative analysis with bar charts and box plots
- Summary statistics and KPI tiles
- Interactive filters for data exploration

## Installation and Setup

### Prerequisites
- Python 3.8+
- Required packages: streamlit, pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, scipy

### Installation
1. Clone or download the dashboard files
2. Install required packages:
```
pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn scipy
```

### Running the Dashboard
Navigate to the dashboard directory and run:
```
streamlit run app.py
```

## Dashboard Structure

### Data
The dashboard uses merged data from `data/merged_ceo_financial_data.csv` sourced from WRDS (ExecuComp + Compustat).

### Components
1. **Interactive Filters (Sidebar)**
   - Filters for Company Ticker, Industry, Year Range, Firm Size, Executive Title, Gender, and Compensation Range.

2. **Summary Statistics & KPIs** (Displayed at the top)
   - Key metrics cards (e.g., Average CEO Pay, ROA for Top-Paid Firms, Alignment Score, Company/Executive Counts).
   - Tabs for "Top Paid Executives" table and "Pay-Performance Scorecard".

3. **Time Series Analysis** (Tab 1)
   - Line chart comparing average CEO pay and a selected performance metric over time.

4. **Scatter Plot Analysis** (Tab 2)
   - Scatter plot of CEO Pay vs. a selected performance metric.
   - Options to color points and filter by firm size (assets).
   - Includes regression line statistics and highlights "pay-for-no-performance" outliers.

5. **Correlation Analysis** (Tab 3)
   - Interactive heatmap showing correlations between selected financial and executive metrics.
   - Option to filter by industry.
   - Includes an "Industry Comparison" tab (currently noted as not implemented).

6. **Comparative Analysis** (Tab 4)
   - Bar chart of average compensation by industry.
   - Line chart showing compensation trends over time.
   - Bar chart comparing average compensation across ROA performance quartiles.

## Usage Guide

### Filtering Data
Use the sidebar filters to narrow down the dataset based on:
- Industry
- Year range
- Executive title
- Gender
- Firm size
- Pay range

Click "Apply Filters" to update all visualizations with the filtered data.

### Exploring Visualizations
Navigate between different analysis types using the tabs:
- **Time Series Analysis**: Select different performance metrics to compare with CEO pay over time
- **Scatter Plot Analysis**: Change the Y-axis metric and color points by different attributes
- **Correlation Analysis**: Select variables to include in the heatmap or compare across industries
- **Comparative Analysis**: Group data by different categories and explore distributions

### Interpreting Results
Each visualization includes an "About this visualization" expander with detailed explanations of what the chart shows and how to interpret it.

## Data Dictionary

### Executive Metrics
- **tdc1**: Total Direct Compensation (salary, bonus, stock options, etc.)
- **gender**: Executive gender
- **age**: Executive age
- **title**: Executive title/position
- **tenure**: Years of service as executive

### Financial Metrics
- **roa**: Return on Assets (%)
- **ni**: Net Income
- **revt**: Revenue
- **at**: Total Assets
- **mkvalt**: Market Value
- **revenue_growth**: Year-over-year revenue growth (%)

## Troubleshooting
- If visualizations don't update after applying filters, try clicking "Reset Filters" and then apply your filters again
- For performance issues with large datasets, try narrowing your filter criteria
- If the dashboard fails to load, ensure all required packages are installed correctly

## Credits
- Contributors: Aastha Surana, Kyle Chin, Matthew Maslow
- Data Source: WRDS (ExecuComp + Compustat) for U.S. companies from 2010-2024
- Last Updated: April 2025
