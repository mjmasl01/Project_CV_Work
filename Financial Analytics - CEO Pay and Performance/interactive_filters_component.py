import streamlit as st
import pandas as pd
import numpy as np

# Load the data
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
    
    return data

# Function to create filters sidebar
def create_filters_sidebar():
    st.sidebar.header("Dashboard Filters")
    
    # Load data
    data = load_data()
    
    # Initialize filtered_data for dynamic filtering
    filtered_data = data.copy()
    
    # Create filter containers with expanders
    with st.sidebar.expander("Company Filters", expanded=True):
        # Company ticker filter
        tickers = ['All Tickers'] + sorted(data['ticker'].unique().tolist())
        selected_ticker = st.selectbox(
            "Company Ticker",
            options=tickers,
            key="filter_ticker"
        )
        
        # Filter data based on ticker selection for dynamic filtering
        if selected_ticker != 'All Tickers':
            filtered_data = filtered_data[filtered_data['ticker'] == selected_ticker]
        
        # Industry filter - dynamically updated based on ticker selection
        available_industries = ['All Industries'] + sorted(filtered_data['industry_name'].unique().tolist())
        selected_industry = st.selectbox(
            "Industry",
            options=available_industries,
            key="filter_industry"
        )
        
        # Further filter data based on industry selection
        if selected_industry != 'All Industries':
            filtered_data = filtered_data[filtered_data['industry_name'] == selected_industry]
        
        # Year range filter - dynamically updated based on previous selections
        if len(filtered_data) > 0:
            min_year = int(filtered_data['year'].min())
            max_year = int(filtered_data['year'].max())
        else:
            min_year = int(data['year'].min())
            max_year = int(data['year'].max())
            
        year_range = st.slider(
            "Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            step=1,
            key="filter_year_range"
        )
        
        # Further filter data based on year selection
        filtered_data = filtered_data[(filtered_data['year'] >= year_range[0]) & 
                                     (filtered_data['year'] <= year_range[1])]
        
        # Firm size filter - dynamically updated based on previous selections
        available_firm_sizes = ['All Sizes']
        if len(filtered_data) > 0:
            available_firm_sizes += sorted(filtered_data['firm_size'].unique().tolist())
        else:
            available_firm_sizes += sorted(data['firm_size'].unique().tolist())
            
        selected_firm_size = st.selectbox(
            "Firm Size (based on total assets)",
            options=available_firm_sizes,
            key="filter_firm_size"
        )
        
        # Further filter data based on firm size selection
        if selected_firm_size != 'All Sizes':
            filtered_data = filtered_data[filtered_data['firm_size'] == selected_firm_size]
    
    with st.sidebar.expander("Executive Filters", expanded=True):
        # Executive title filter - dynamically updated based on previous selections
        if len(filtered_data) > 0:
            title_counts = filtered_data['title'].value_counts().head(10)
        else:
            title_counts = data['title'].value_counts().head(10)
            
        available_titles = ['All Titles'] + title_counts.index.tolist()
        
        selected_title = st.selectbox(
            "Executive Title",
            options=available_titles,
            key="filter_title"
        )
        
        # Further filter data based on title selection
        if selected_title != 'All Titles':
            filtered_data = filtered_data[filtered_data['title'] == selected_title]
        
        # Gender filter - dynamically updated based on previous selections
        if len(filtered_data) > 0:
            available_genders = ['All'] + sorted(filtered_data['gender'].unique().tolist())
        else:
            available_genders = ['All'] + sorted(data['gender'].unique().tolist())
            
        selected_gender = st.selectbox(
            "Gender",
            options=available_genders,
            key="filter_gender"
        )
        
        # Further filter data based on gender selection
        if selected_gender != 'All':
            filtered_data = filtered_data[filtered_data['gender'] == selected_gender]
    
        # Pay range filter - dynamically updated based on previous selections
        if len(filtered_data) > 0:
            min_pay = float(filtered_data['tdc1'].min())
            max_pay = float(filtered_data['tdc1'].quantile(0.99) if len(filtered_data) > 10 else filtered_data['tdc1'].max())
        else:
            min_pay = float(data['tdc1'].min())
            max_pay = float(data['tdc1'].quantile(0.99))
            
        pay_range = st.slider(
            "Annual Compensation Range",
            min_value=min_pay/1000,
            max_value=max_pay/1000,
            value=(min_pay/1000, max_pay/1000),
            step=100.0,
            format="$%.0fM",
            key="filter_pay_range"
        )
        
        # Convert back to actual values
        pay_range = (pay_range[0]*1000 , pay_range[1]*1000)

    #with st.sidebar.expander("Financial Filters", expanded=True):

        
        # Performance metric filter
        # performance_metrics = {
        #     'roa': 'Return on Assets (ROA) %',
        #     'ni': 'Net Income ($)',
        #     'revt': 'Revenue ($)',
        #     'revenue_growth': 'Revenue Growth (%)'
        # }
        
        # selected_metric = st.selectbox(
        #     "Performance Metric",
        #     options=list(performance_metrics.keys()),
        #     format_func=lambda x: performance_metrics[x],
        #     key="filter_performance_metric"
        # )
    
    # Create a button to apply filters
    apply_filters = st.sidebar.button("Apply Filters", key="apply_filters_button")
    
    # Create a button to reset filters
    reset_filters = st.sidebar.button("Reset Filters", key="reset_filters_button")
    
    # Return filter values
    filters = {
        'ticker': selected_ticker,
        'industry': selected_industry,
        'year_range': year_range,
        'firm_size': selected_firm_size,
        'title': selected_title,
        'gender': selected_gender,
        'pay_range': pay_range,
        #'performance_metric': selected_metric,
        'apply': apply_filters,
        'reset': reset_filters
    }
    
    return filters

# Function to apply filters to data
def apply_filters_to_data(data, filters):
    filtered_data = data.copy()
    
    # Apply ticker filter
    if filters['ticker'] != 'All Tickers':
        filtered_data = filtered_data[filtered_data['ticker'] == filters['ticker']]
    
    # Apply industry filter
    if filters['industry'] != 'All Industries':
        filtered_data = filtered_data[filtered_data['industry_name'] == filters['industry']]
    
    # Apply year range filter
    filtered_data = filtered_data[(filtered_data['year'] >= filters['year_range'][0]) & 
                                 (filtered_data['year'] <= filters['year_range'][1])]
    
    # Apply firm size filter
    if filters['firm_size'] != 'All Sizes':
        filtered_data = filtered_data[filtered_data['firm_size'] == filters['firm_size']]
    
    # Apply title filter
    if filters['title'] != 'All Titles':
        filtered_data = filtered_data[filtered_data['title'] == filters['title']]
    
    # Apply gender filter
    if filters['gender'] != 'All':
        filtered_data = filtered_data[filtered_data['gender'] == filters['gender']]
    
    # Apply pay range filter
    filtered_data = filtered_data[(filtered_data['tdc1'] >= filters['pay_range'][0]) & 
                                 (filtered_data['tdc1'] <= filters['pay_range'][1])]
    
    return filtered_data

# Function to display filter summary
def display_filter_summary(filters):
    st.markdown("### Current Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Company Ticker:** {filters['ticker']}")
        st.markdown(f"**Industry:** {filters['industry']}")
        st.markdown(f"**Year Range:** {filters['year_range'][0]} - {filters['year_range'][1]}")
    
    with col2:
        st.markdown(f"**Firm Size:** {filters['firm_size']}")
        st.markdown(f"**Executive Title:** {filters['title']}")
        st.markdown(f"**Gender:** {filters['gender']}")
    
    with col3:
        st.markdown(f"**Annual Pay Range:** ${filters['pay_range'][0]/1000:.0f}K - ${filters['pay_range'][1]/1000:.0f}K")
        
        # Map performance metric to display name
        performance_metrics = {
            'roa': 'Return on Assets (ROA) %',
            'ni': 'Net Income ($)',
            'revt': 'Revenue ($)',
            'revenue_growth': 'Revenue Growth (%)'
        }
        st.markdown(f"**Performance Metric:** {performance_metrics[filters['performance_metric']]}")

# Main function to run the filters component
def interactive_filters_component():
    # Create filters sidebar
    filters = create_filters_sidebar()
    
    # Load data
    data = load_data()
    
    # Apply filters to data
    if filters['apply']:
        filtered_data = apply_filters_to_data(data, filters)
        st.session_state['filtered_data'] = filtered_data
        st.session_state['filters'] = filters
    elif filters['reset']:
        st.session_state['filtered_data'] = data
        st.session_state['filters'] = filters
    elif 'filtered_data' not in st.session_state:
        st.session_state['filtered_data'] = data
        st.session_state['filters'] = filters
    
    # Display filter summary
    display_filter_summary(st.session_state['filters'])
    
    # Display data count
    st.markdown(f"**Filtered Data: {len(st.session_state['filtered_data']):,} records**")
    
    # Return filtered data for use in other components
    return st.session_state['filtered_data']

if __name__ == "__main__":
    interactive_filters_component()
