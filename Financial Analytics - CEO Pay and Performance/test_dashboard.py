import streamlit as st
import pandas as pd
import numpy as np
import os

# Function to load data
@st.cache_data
def load_data():
    data = pd.read_csv('data/merged_ceo_financial_data.csv')
    # Clean data
    data = data.dropna(subset=['tdc1', 'roa', 'ni', 'revt'])
    # Convert year to int
    data['year'] = data['year'].astype(int)
    # Create SIC industry mapping
    data['industry'] = data['sic'].astype(str).str[:2]
    # Map common SIC codes to industry names
    industry_map = {
        '73': 'Business Services',
        '36': 'Electronics',
        '28': 'Chemicals',
        '35': 'Industrial Machinery',
        '38': 'Instruments',
        '37': 'Transportation Equipment',
        '48': 'Communications',
        '49': 'Utilities',
        '20': 'Food Products',
        '33': 'Primary Metal Industries'
    }
    data['industry_name'] = data['industry'].map(industry_map).fillna('Other')
    
    # Calculate firm size categories based on assets
    data['firm_size'] = pd.qcut(data['at'], 4, labels=['Small', 'Medium', 'Large', 'Very Large'])
    
    # Clean revenue growth (remove inf values)
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # Calculate executive tenure based on years in dataset
    tenure_data = data.groupby('execid')['year'].agg(['min', 'max'])
    tenure_data['tenure'] = tenure_data['max'] - tenure_data['min'] + 1
    data = pd.merge(data, tenure_data['tenure'], left_on='execid', right_index=True, how='left')
    
    # Calculate performance quartiles
    data['roa_quartile'] = pd.qcut(data['roa'], 4, labels=['Bottom 25%', 'Lower Middle', 'Upper Middle', 'Top 25%'])
    
    return data

# Function to test data loading
def test_data_loading():
    try:
        data = load_data()
        print(f"Data loaded successfully. Shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        print(f"Year range: {data['year'].min()} - {data['year'].max()}")
        print(f"Number of companies: {data['gvkey'].nunique()}")
        print(f"Number of executives: {data['execid'].nunique()}")
        print(f"Number of CEOs: {len(data[data['ceoann'] == 'CEO'])}")
        print(f"Average CEO pay: ${data[data['ceoann'] == 'CEO']['tdc1'].mean():,.2f}")
        print(f"Average ROA: {data['roa'].mean():.2f}%")
        print("Data loading test passed!")
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

# Function to test component imports
def test_component_imports():
    try:
        import time_series_component
        import scatter_plot_component
        import correlation_heatmap_component
        import interactive_filters_component
        import bar_charts_box_plots_component
        import summary_stats_kpi_component
        print("All components imported successfully!")
        return True
    except Exception as e:
        print(f"Error importing components: {e}")
        return False

# Function to test dashboard functionality
def test_dashboard():
    print("Testing dashboard functionality...")
    
    # Test data loading
    data_test = test_data_loading()
    
    # Test component imports
    import_test = test_component_imports()
    
    # Check if all tests passed
    if data_test and import_test:
        print("All tests passed! Dashboard is ready to run.")
        return True
    else:
        print("Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    test_dashboard()
