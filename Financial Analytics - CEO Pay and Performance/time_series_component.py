import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    return data

# Function to create time series visualization
def create_time_series_chart(data, metric='roa'):
    # Group by year
    yearly_pay = data.groupby('year')['tdc1'].mean().reset_index()
    
    if metric == 'roa':
        yearly_metric = data.groupby('year')['roa'].mean().reset_index()
        metric_label = 'Return on Assets (%)'
    elif metric == 'ni':
        yearly_metric = data.groupby('year')['ni'].mean().reset_index()
        metric_label = 'Net Income'
    elif metric == 'revt':
        yearly_metric = data.groupby('year')['revt'].mean().reset_index()
        metric_label = 'Revenue'
    elif metric == 'revenue_growth':
        # Filter out infinite values
        temp_data = data[~data['revenue_growth'].isin([np.inf, -np.inf])]
        temp_data = temp_data.dropna(subset=['revenue_growth'])
        yearly_metric = temp_data.groupby('year')['revenue_growth'].mean().reset_index()
        metric_label = 'Revenue Growth (%)'
    
    # Create subplot with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add CEO pay trace
    fig.add_trace(
        go.Scatter(
            x=yearly_pay['year'],
            y=yearly_pay['tdc1'],
            name="CEO Compensation",
            line=dict(color='blue', width=2)
        ),
        secondary_y=False,
    )
    
    # Add performance metric trace
    fig.add_trace(
        go.Scatter(
            x=yearly_metric['year'],
            y=yearly_metric[metric],
            name=metric_label,
            line=dict(color='red', width=2)
        ),
        secondary_y=True,
    )
    
    # Set titles
    fig.update_layout(
        title_text="CEO Pay and Performance Over Time",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Average CEO Compensation", secondary_y=False)
    fig.update_yaxes(title_text=metric_label, secondary_y=True)
    
    # Calculate year-over-year alignment
    if len(yearly_pay) > 1 and len(yearly_metric) > 1:
        pay_change = yearly_pay['tdc1'].pct_change().dropna()
        metric_change = yearly_metric[metric].pct_change().dropna()
        
        # Ensure both series have the same index
        common_years = set(pay_change.index).intersection(set(metric_change.index))
        pay_change = pay_change.loc[list(common_years)]
        metric_change = metric_change.loc[list(common_years)]
        
        # Calculate alignment (positive when both move in same direction)
        alignment = (pay_change * metric_change > 0).astype(int)
        alignment_years = yearly_pay['year'].iloc[list(common_years)]
        
        # Add alignment markers
        for i, year in enumerate(alignment_years):
            if i < len(alignment):
                marker_color = 'green' if alignment.iloc[i] == 1 else 'red'
                fig.add_trace(
                    go.Scatter(
                        x=[year],
                        y=[yearly_pay['tdc1'].iloc[i+1]],
                        mode='markers',
                        marker=dict(color=marker_color, size=12, symbol='star'),
                        name=f"{'Aligned' if alignment.iloc[i] == 1 else 'Misaligned'}",
                        showlegend=i==0  # Only show one legend item for alignment
                    ),
                    secondary_y=False,
                )
    
    return fig

# Main function to run the time series visualization component
def time_series_component(filtered_data=None):
    st.header("Time Series Analysis")
    
    # Load data if filtered_data is not provided
    if filtered_data is None:
        # Load data
        data = load_data()
        
        # Use filtered data if available in session state
        if 'filtered_data' in st.session_state:
            filtered_data = st.session_state['filtered_data']
        else:
            filtered_data = data
    
    # Check if filtered data is empty or has only one year
    if len(filtered_data) == 0:
        st.warning("No data available for time series analysis with current filters.")
        return
    
    if filtered_data['year'].nunique() < 2:
        st.warning("Time series analysis requires data from at least two years. Please adjust your filters.")
        return
    
    # Create metric selector
    metric_options = {
        #'roa': 'Return on Assets (ROA)',
        'ni': 'Net Income',
        #'revt': 'Revenue',
        #'revenue_growth': 'Revenue Growth'
    }
    
    # Get performance metric from session state if available
    if 'filters' in st.session_state and 'performance_metric' in st.session_state['filters']:
        default_metric = st.session_state['filters']['performance_metric']
    else:
        default_metric = 'ni'
    
    selected_metric = default_metric
    # selected_metric = st.selectbox(
    #     "Select Performance Metric",
    #     options=list(metric_options.keys()),
    #     format_func=lambda x: metric_options[x],
    #     index=list(metric_options.keys()).index(default_metric),
    #     key="time_series_metric"
    # )
    
    # Create and display the time series chart
    fig = create_time_series_chart(filtered_data, selected_metric)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    with st.expander("About this visualization"):
        st.markdown("""
        This chart shows the relationship between average CEO compensation and company performance over time.
        
        - **Blue line**: Average CEO compensation (tdc1) for each year
        - **Red line**: Average performance metric for each year
        - **Star markers**: Indicate alignment (green) or misalignment (red) between pay and performance changes
        
        Alignment occurs when both CEO pay and the performance metric move in the same direction year-over-year.
        """)

if __name__ == "__main__":
    time_series_component()
