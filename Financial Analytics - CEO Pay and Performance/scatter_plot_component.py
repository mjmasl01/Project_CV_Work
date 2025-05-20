import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from scipy import stats

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
    
    return data

# Function to create scatter plot with trendline
def create_scatter_plot(data, x_col='tdc1', y_col='roa', color_col=None):
    # Remove NaN values
    plot_data = data.dropna(subset=[x_col, y_col])
    
    # Create scatter plot
    if color_col:
        fig = px.scatter(
            plot_data, 
            x=x_col, 
            y=y_col, 
            color=color_col,
            opacity=0.7,
            trendline="ols",
            trendline_color_override="red",
            hover_data=['coname', 'year', 'tdc1', 'roa', 'ni', 'revt']
        )
    else:
        fig = px.scatter(
            plot_data, 
            x=x_col, 
            y=y_col,
            opacity=0.7,
            trendline="ols",
            trendline_color_override="red",
            hover_data=['coname', 'year', 'tdc1', 'roa', 'ni', 'revt']
        )
    
    # Update layout
    fig.update_layout(
        title=f"CEO Compensation vs {y_col.upper() if y_col in ['roa', 'ni'] else y_col.title()}",
        xaxis_title="CEO Compensation (tdc1)",
        yaxis_title=y_col.upper() if y_col in ['roa', 'ni'] else y_col.title(),
        height=600
    )
    
    # Identify outliers - "pay-for-no-performance" cases
    # High pay (top 25%) but low performance (bottom 25%)
    high_pay_threshold = plot_data[x_col].quantile(0.75)
    low_perf_threshold = plot_data[y_col].quantile(0.25)
    
    outliers = plot_data[(plot_data[x_col] > high_pay_threshold) & 
                         (plot_data[y_col] < low_perf_threshold)]
    
    # Add outliers as separate trace
    if not outliers.empty:
        fig.add_trace(
            go.Scatter(
                x=outliers[x_col],
                y=outliers[y_col],
                mode='markers',
                marker=dict(
                    color='red',
                    size=12,
                    line=dict(
                        color='black',
                        width=2
                    )
                ),
                name="Pay-for-No-Performance Outliers",
                text=outliers['coname'] + " (" + outliers['year'].astype(str) + ")",
                hoverinfo="text"
            )
        )
    
    # Calculate and display regression statistics
    X = plot_data[x_col].values.reshape(-1, 1)
    y = plot_data[y_col].values
    
    # Fit regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate R-squared
    r_squared = model.score(X, y)
    
    # Calculate p-value
    n = len(X)
    p = 1  # Number of predictors
    t_stat = model.coef_[0] / (np.sqrt((1 - r_squared) / (n - p - 1)) / np.sqrt(np.sum((X - np.mean(X))**2)))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - p - 1))
    
    # Add annotation with regression statistics
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.98,  # Move to top-right
        xanchor='right', # Anchor to the right
        text=f"R² = {r_squared:.3f}<br>p-value = {p_value:.3e}<br>Linear Regression Slope = {model.coef_[0]:.3f}",
        showarrow=False,
        align='left',    # Ensure left alignment within the box
        font=dict(size=15, color="black"), # Slightly larger font
        bgcolor="rgba(255, 255, 255, 0.9)", # More opaque background
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
    )
    return fig, outliers

# Main function to run the scatter plot component
def scatter_plot_component(filtered_data=None):
    st.header("Scatter Plot Analysis")
    
    # Load data if filtered_data is not provided
    if filtered_data is None:
        # Load data
        data = load_data()
        
        # Use filtered data if available in session state
        if 'filtered_data' in st.session_state:
            filtered_data = st.session_state['filtered_data']
        else:
            filtered_data = data
    
    # Check if filtered data is empty
    if len(filtered_data) == 0:
        st.warning("No data available for scatter plot analysis with current filters.")
        return
    
    # Create columns for controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Create y-axis metric selector
        metric_options = {
            'roa': 'Return on Assets (ROA) %',
            'ni': 'Net Income ($)',
            'revt': 'Revenue ($)',
            'revenue_growth': 'Revenue Growth (%)'
        }
        
        # Get performance metric from session state if available
        if 'filters' in st.session_state and 'performance_metric' in st.session_state['filters']:
            default_metric = st.session_state['filters']['performance_metric']
        else:
            default_metric = 'roa'
        
        selected_metric = st.selectbox(
            "Select Performance Metric (Y-axis)",
            options=list(metric_options.keys()),
            format_func=lambda x: metric_options[x],
            index=list(metric_options.keys()).index(default_metric),
            key="scatter_metric"
        )
    
    with col2:
        # Create color-by selector
        color_options = {
            None: 'None',
            'industry_name': 'Industry',
            'firm_size': 'Firm Size',
            'gender': 'Gender',
            'year': 'Year'
        }
        
        selected_color = st.selectbox(
            "Color Points By",
            options=list(color_options.keys()),
            format_func=lambda x: color_options[x] if x is not None else 'None',
            key="scatter_color"
        )
    
    with col3:
        # Create filter for minimum firm size (assets)
        if len(filtered_data) > 0:
            min_assets = st.slider(
                "Minimum Firm Size (Assets in $)",
                min_value=float(filtered_data['at'].min()),
                max_value=float(filtered_data['at'].max()),
                value=float(filtered_data['at'].min()),
                format="$%.0f",
                key="scatter_min_assets"
            )
        else:
            min_assets = 0
    
    # Filter data based on minimum assets
    scatter_filtered_data = filtered_data[filtered_data['at'] >= min_assets]
    
    # Check if scatter_filtered_data is empty after additional filtering
    if len(scatter_filtered_data) == 0:
        st.warning("No data available after applying minimum assets filter. Please adjust your filters.")
        return
    
    # Create and display the scatter plot
    fig, outliers = create_scatter_plot(scatter_filtered_data, 'tdc1', selected_metric, selected_color)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display outliers table
    if not outliers.empty:
        with st.expander("Pay-for-No-Performance Outliers"):
            st.write(f"Found {len(outliers)} companies with high pay but low {metric_options[selected_metric]}:")
            st.dataframe(
                outliers[['coname', 'year', 'tdc1', selected_metric, 'industry_name']].sort_values('tdc1', ascending=False),
                hide_index=True
            )
    
    # Add explanation
    with st.expander("About this visualization"):
        st.markdown("""
        This scatter plot shows the relationship between CEO compensation and company performance.
        
        - Each point represents a company-year observation
        - The red trendline shows the overall relationship between pay and performance
        - Red highlighted points are "pay-for-no-performance" outliers (high pay, low performance)
        - R² value indicates how much of the variation in performance is explained by CEO pay
        - p-value indicates statistical significance of the relationship
        
        You can change the performance metric, color the points by different attributes, and filter by firm size.
        """)

if __name__ == "__main__":
    scatter_plot_component()
