import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

# Function to create KPI metric card
def create_metric_card(title, value, delta=None, delta_suffix="vs. previous period", is_currency=False, is_percentage=False):
    if is_currency:
        formatted_value = f"${value:,.0f}"
        if delta is not None:
            delta_prefix = "+" if delta > 0 else ""
            formatted_delta = f"{delta_prefix}${delta:,.0f} {delta_suffix}"
    elif is_percentage:
        formatted_value = f"{value:.2f}%"
        if delta is not None:
            delta_prefix = "+" if delta > 0 else ""
            formatted_delta = f"{delta_prefix}{delta:.2f}% {delta_suffix}"
    else:
        formatted_value = f"{value:,.2f}"
        if delta is not None:
            delta_prefix = "+" if delta > 0 else ""
            formatted_delta = f"{delta_prefix}{delta:,.2f} {delta_suffix}"
    
    if delta is not None:
        st.metric(
            label=title,
            value=formatted_value,
            delta=formatted_delta,
            delta_color="normal"
        )
    else:
        st.metric(
            label=title,
            value=formatted_value
        )

# Function to calculate average CEO pay
def calculate_avg_ceo_pay(data):
    return data['tdc1'].mean()

# Function to get top paid executives
def get_top_paid_executives(data, n=5):
    return data.sort_values('tdc1', ascending=False).head(n)[['coname', 'title', 'year', 'tdc1', 'gender']]

# Function to calculate average ROA for top-paid firms
def calculate_avg_roa_top_paid(data, top_pct=0.25):
    # Get top 25% of firms by CEO pay
    top_paid_threshold = data['tdc1'].quantile(1 - top_pct)
    top_paid_firms = data[data['tdc1'] >= top_paid_threshold]
    return top_paid_firms['roa'].mean()

# Function to calculate pay-performance alignment score
def calculate_pay_performance_alignment(data):
    # Calculate correlation between pay and performance
    pay_roa_corr = data[['tdc1', 'roa']].corr().iloc[0, 1]
    
    # Scale to a 0-100 score (0 = negative correlation, 50 = no correlation, 100 = perfect positive correlation)
    alignment_score = (pay_roa_corr + 1) * 50
    
    return alignment_score

# Function to create pay-performance scorecard
def create_pay_performance_scorecard(data):
    # Calculate metrics for scorecard
    metrics = {}
    
    # Pay-performance correlation
    metrics['pay_roa_corr'] = data[['tdc1', 'roa']].corr().iloc[0, 1]
    metrics['pay_ni_corr'] = data[['tdc1', 'ni']].corr().iloc[0, 1]
    metrics['pay_revt_corr'] = data[['tdc1', 'revt']].corr().iloc[0, 1]
    
    # Percentage of firms with high pay but low performance
    high_pay_threshold = data['tdc1'].quantile(0.75)
    low_perf_threshold = data['roa'].quantile(0.25)
    metrics['pct_high_pay_low_perf'] = len(data[(data['tdc1'] > high_pay_threshold) & (data['roa'] < low_perf_threshold)]) / len(data) * 100
    
    # Percentage of firms with aligned pay and performance
    data['pay_rank'] = data['tdc1'].rank(pct=True)
    data['roa_rank'] = data['roa'].rank(pct=True)
    data['pay_perf_diff'] = (data['pay_rank'] - data['roa_rank']).abs()
    metrics['pct_aligned'] = len(data[data['pay_perf_diff'] < 0.25]) / len(data) * 100
    
    return metrics

# Main function to run the summary statistics and KPI tiles component
def summary_stats_kpi_component(filtered_data=None):
    st.header("Summary Statistics & Key Performance Indicators")
    
    # Load data if filtered_data is not provided
    if filtered_data is None:
        # Load data
        data = load_data()
        
        # Use filtered data if available in session state
        if 'filtered_data' in st.session_state:
            filtered_data = st.session_state['filtered_data']
        else:
            filtered_data = data
    else:
        # Load full data for comparison
        data = load_data()
    
    # Create KPI metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Average CEO Pay
        avg_pay = calculate_avg_ceo_pay(filtered_data)
        
        # Calculate delta vs all data if filtered
        if len(filtered_data) < len(data):
            avg_pay_all = calculate_avg_ceo_pay(data)
            
            #delta_pay = avg_pay - avg_pay_all
            #delta_suffix = "vs. overall average"
        else:
            # If not filtered, calculate delta vs previous year
            current_year = filtered_data['year'].max()
            prev_year = current_year - 1
            current_year_data = filtered_data[filtered_data['year'] == current_year]
            prev_year_data = filtered_data[filtered_data['year'] == prev_year]
            
            if not current_year_data.empty and not prev_year_data.empty:
                avg_pay_current = calculate_avg_ceo_pay(current_year_data)
                avg_pay_prev = calculate_avg_ceo_pay(prev_year_data)
                #delta_pay = avg_pay_current - avg_pay_prev
                #delta_suffix = f"vs. {prev_year}"
            else:
                delta_pay = None
                delta_suffix = ""
        
        create_metric_card("Average CEO Pay (in '000)", avg_pay, is_currency=True)
    
    with col2:
        # Average ROA for top-paid firms
        avg_roa_top = calculate_avg_roa_top_paid(filtered_data)
        
        # Calculate delta vs all firms
        avg_roa_all = filtered_data['roa'].mean()
        delta_roa = avg_roa_top - avg_roa_all
        
        create_metric_card("Avg ROA for Top-Paid Firms", avg_roa_top, delta_roa, "vs. all firms", is_percentage=True)
    
    with col3:
        # Pay-Performance Alignment Score
        alignment_score = calculate_pay_performance_alignment(filtered_data)
        
        # Calculate delta vs all data if filtered
        if len(filtered_data) < len(data):
            alignment_score_all = calculate_pay_performance_alignment(data)
            delta_alignment = alignment_score - alignment_score_all
            delta_suffix = "vs. overall alignment"
        else:
            delta_alignment = None
            delta_suffix = ""
        
        create_metric_card("Pay-Performance Alignment Score", alignment_score, delta_alignment, delta_suffix)
    
    with col4:
        # Number of companies and executives
        num_companies = filtered_data['gvkey'].nunique()
        num_executives = filtered_data['execid'].nunique()
        
        st.metric(
            label="Number of Companies",
            value=f"{num_companies:,}",
            delta=f"{num_executives:,} Executives",
        )
    
    # Create tabs for detailed statistics
    tab1, tab2 = st.tabs(["Top Paid Executives", "Pay-Performance Scorecard"])
    
    with tab1:
        # Get top paid executives
        top_execs = get_top_paid_executives(filtered_data)
        
        # Format the table
        top_execs_display = top_execs.copy()
        top_execs_display['tdc1'] = top_execs_display['tdc1'].apply(lambda x: f"${x:,.0f}")
        top_execs_display.columns = ['Company', 'Title', 'Year', 'Compensation (in \'000)', 'Gender']
        
        # Display the table
        st.subheader("Top 5 Highest Paid Executives")
        st.dataframe(top_execs_display, hide_index=True, use_container_width=True)

        with st.expander("About these metrics"):
            st.markdown("""
        ### Key Performance Indicators
        
        - **Average CEO Pay**: The mean total compensation (tdc1) for CEOs in the filtered dataset.
        - **Avg ROA for Top-Paid Firms**: The mean Return on Assets for companies with CEO compensation in the top 25%.
        - **Pay-Performance Alignment Score**: A score from 0-100 indicating how well CEO pay aligns with company performance (ROA).
          - 0 = Perfect negative correlation (higher pay associated with lower performance)
          - 50 = No correlation
          - 100 = Perfect positive correlation (higher pay associated with higher performance)
        - **Number of companies**: The number of unique companies and executives in the filtered dataset.
        
        ### Top Paid Executives
        
        This table shows the 5 highest-paid executives in the filtered dataset, including their company, title, year, and compensation.
        
            """)
    
    with tab2:
        # Create pay-performance scorecard
        scorecard_metrics = create_pay_performance_scorecard(filtered_data)
        
        # Create columns for scorecard
        sc_col1, sc_col2 = st.columns(2)
        
        with sc_col1:
            st.subheader("Pay-Performance Correlations")
            
            # Create gauge charts for correlations
            fig = go.Figure()
            
            # Add ROA correlation gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=float("{:.2f}".format(scorecard_metrics['pay_roa_corr'])),
                title={'text': "Pay vs. ROA"},
                domain={'x': [0.2, 0.9], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, 0], 'color': "red"},
                        #{'range': [-0.5, 0], 'color': "orange"},
                        #{'range': [0, 0.5], 'color': "lightgreen"},
                        {'range': [0, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': scorecard_metrics['pay_roa_corr']
                    }
                }
            ))
            
            # Update layout
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add other correlations as metrics
            st.metric("Pay vs. Net Income Correlation", f"{scorecard_metrics['pay_ni_corr']:.2f}")
            st.metric("Pay vs. Revenue Correlation", f"{scorecard_metrics['pay_revt_corr']:.2f}")
            
        with st.expander("About these metrics"):
            st.markdown("""     ### Pay-Performance Scorecard
        
        - **Pay-Performance Correlations**: The correlation coefficients between CEO pay and various performance metrics.
            - Values range from -1 (perfect negative correlation) to 1 (perfect positive correlation).
            - Values close to 0 indicate little to no correlation.
        
        - **Pay-Performance Alignment**: The percentage of companies where CEO pay rank (percentile) is within 25 percentage points of their performance rank.
            - If the CEO’s pay rank and performance rank are within 25 percentile points of each other (up or down), it is considered "aligned."
            - Example: If a CEO’s pay rank is at the 80th percentile and the company’s performance is at the 65th percentile → difference is 15 points → considered aligned.
        
        - **High Pay, Low Performance**: The percentage of companies with high CEO pay (top 25%) but low performance (bottom 25%).""")
        with sc_col2:
            st.subheader("Pay-Performance Alignment")
            
            # Create a donut chart for alignment percentage
            fig = go.Figure()
            
            # Add aligned percentage
            fig.add_trace(go.Pie(
                labels=["Aligned", "Misaligned"],
                values=[scorecard_metrics['pct_aligned'], 100 - scorecard_metrics['pct_aligned']],
                hole=0.7,
                marker_colors=['green', 'red']
            ))
            
            # Add annotation in the center
            fig.update_layout(
                annotations=[dict(
                    text=f"{scorecard_metrics['pct_aligned']:.1f}%<br>Aligned",
                    x=0.5, y=0.5,
                    font_size=20,
                    showarrow=False
                )],
                height=250,
                margin=dict(l=20, r=20, t=30, b=20),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.1,
                    xanchor="center",
                    x=0.5
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add high pay, low performance metric
            st.metric(
                "High Pay, Low Performance",
                f"{scorecard_metrics['pct_high_pay_low_perf']:.1f}%",
                #delta=f"{len(filtered_data[(filtered_data['tdc1'] > filtered_data['tdc1'].quantile(0.75)) & (filtered_data['roa'] < filtered_data['roa'].quantile(0.25))]):,} companies",
                #delta_color="inverse"  # Lower is better for this metric
            )
    


if __name__ == "__main__":
    summary_stats_kpi_component()
