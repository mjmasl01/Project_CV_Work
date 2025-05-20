import streamlit as st
import pandas as pd
import numpy as np
import os

# Import components
from time_series_component import time_series_component
from scatter_plot_component import scatter_plot_component
from correlation_heatmap_component import correlation_heatmap_component
from interactive_filters_component import interactive_filters_component, create_filters_sidebar, apply_filters_to_data, load_data
from simplified_bar_charts_component import bar_charts_box_plots_component
from summary_stats_kpi_component import summary_stats_kpi_component

# Set page configuration
st.set_page_config(
    page_title="CEO Pay and Performance Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve dashboard appearance - using Streamlit's native theming approach
st.markdown("""
<style>
    /* Base styling for all themes */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    h1, h2, h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Improved tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid var(--primary-color);
    }
    
    /* Improved metrics styling */
    div.stMetric {
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Layout improvements */
    .metric-row {
        display: flex;
        justify-content: space-between;
    }
    .stExpander {
        border-radius: 5px;
    }
    
    /* Chart improvements */
    .stPlotlyChart {
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-radius: 5px;
        overflow: hidden;
    }
    
    /* Sidebar improvements */
    .sidebar .sidebar-content {
        padding-top: 1rem;
    }
    
    /* Improve slider visibility */
    .stSlider {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Improve selectbox */
    .stSelectbox {
        margin-bottom: 0.5rem;
    }
    
    /* Improve table styling */
    .stTable {
        border-radius: 5px;
        overflow: hidden;
    }
    
    /* Improve warning messages */
    .stAlert {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Dashboard title
    st.title("CEO Pay and Performance Dashboard")
    st.markdown("Contributors: Aastha Surana, Kyle Chin, Matthew Maslow")
    st.markdown("Exploring the relationship between executive compensation and firm performance")
    
    # Initialize session state for filtered data if not exists
    if 'filtered_data' not in st.session_state:
        data = load_data()
        st.session_state['filtered_data'] = data
        st.session_state['filters'] = {}
    
    # Create sidebar filters
    filters = create_filters_sidebar()
    
    # Load data
    data = load_data()
    
    # Apply filters to data
    if filters['apply'] or filters['ticker'] != 'All Tickers':  # Auto-apply when ticker changes
        filtered_data = apply_filters_to_data(data, filters)
        st.session_state['filtered_data'] = filtered_data
        st.session_state['filters'] = filters
    elif filters['reset']:
        st.session_state['filtered_data'] = data
        st.session_state['filters'] = filters
    
    # Get the current filtered data
    filtered_data = st.session_state['filtered_data']
    
    # Check if filtered data is empty
    if len(filtered_data) == 0:
        st.warning("No data matches the selected filters. Please adjust your filter criteria.")
        return
    
    # Display summary statistics and KPI tiles with filtered data
    summary_stats_kpi_component(filtered_data)
    
    # Create tabs for different visualizations
    tab1,tab2, tab3, tab4 = st.tabs([
        "Time Series Analysis", 
        "Scatter Plot Analysis", 
        "Correlation Analysis",
        "Comparative Analysis"
    ])
    
    with tab1:
        time_series_component(filtered_data)
    
    with tab2:
        scatter_plot_component(filtered_data)
    
    with tab3:
        correlation_heatmap_component(filtered_data)
    
    with tab4:
        # Using the simplified bar charts component
        bar_charts_box_plots_component(filtered_data)
    
    # Add footer with data source information
    st.markdown("---")
    st.markdown("""
    **Data Source:** WRDS (ExecuComp + Compustat) for U.S. companies from 2010-2024  
    **Last Updated:** April 2025  
    """)

if __name__ == "__main__":
    main()
