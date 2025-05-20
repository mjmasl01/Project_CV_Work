import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import os

# =====================
# DATA LOADING FUNCTION (Unified)
# =====================
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
    if 'execid' in data.columns:
        tenure_data = data.groupby('execid')['year'].agg(['min', 'max'])
        tenure_data['tenure'] = tenure_data['max'] - tenure_data['min'] + 1
        data = pd.merge(data, tenure_data['tenure'], left_on='execid', right_index=True, how='left')
    # Calculate performance quartiles if not present
    if 'roa' in data.columns and 'roa_quartile' not in data.columns:
        try:
            data['roa_quartile'] = pd.qcut(data['roa'], 4, labels=['Bottom 25%', 'Lower Middle', 'Upper Middle', 'Top 25%'])
        except Exception:
            pass
    return data

# =====================
# FILTERS COMPONENTS
# =====================
def create_filters_sidebar():
    st.sidebar.header("Dashboard Filters")
    data = load_data()
    filtered_data = data.copy()
    with st.sidebar.expander("Company Filters", expanded=True):
        tickers = ['All Tickers'] + sorted(data['ticker'].unique().tolist())
        selected_ticker = st.selectbox("Company Ticker", options=tickers, key="filter_ticker")
        if selected_ticker != 'All Tickers':
            filtered_data = filtered_data[filtered_data['ticker'] == selected_ticker]
        available_industries = ['All Industries'] + sorted(filtered_data['industry_name'].unique().tolist())
        selected_industry = st.selectbox("Industry", options=available_industries, key="filter_industry")
        if selected_industry != 'All Industries':
            filtered_data = filtered_data[filtered_data['industry_name'] == selected_industry]
        if len(filtered_data) > 0:
            min_year = int(filtered_data['year'].min())
            max_year = int(filtered_data['year'].max())
        else:
            min_year = int(data['year'].min())
            max_year = int(data['year'].max())
        year_range = st.slider("Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year), step=1, key="filter_year_range")
        filtered_data = filtered_data[(filtered_data['year'] >= year_range[0]) & (filtered_data['year'] <= year_range[1])]
        available_firm_sizes = ['All Sizes']
        if len(filtered_data) > 0:
            available_firm_sizes += sorted(filtered_data['firm_size'].unique().tolist())
        else:
            available_firm_sizes += sorted(data['firm_size'].unique().tolist())
        selected_firm_size = st.selectbox("Firm Size (based on total assets)", options=available_firm_sizes, key="filter_firm_size")
        if selected_firm_size != 'All Sizes':
            filtered_data = filtered_data[filtered_data['firm_size'] == selected_firm_size]
    with st.sidebar.expander("Executive Filters", expanded=True):
        if len(filtered_data) > 0:
            title_counts = filtered_data['title'].value_counts().head(10)
        else:
            title_counts = data['title'].value_counts().head(10)
        available_titles = ['All Titles'] + title_counts.index.tolist()
        selected_title = st.selectbox("Executive Title", options=available_titles, key="filter_title")
        if selected_title != 'All Titles':
            filtered_data = filtered_data[filtered_data['title'] == selected_title]
        if len(filtered_data) > 0:
            available_genders = ['All'] + sorted(filtered_data['gender'].unique().tolist())
        else:
            available_genders = ['All'] + sorted(data['gender'].unique().tolist())
        selected_gender = st.selectbox("Gender", options=available_genders, key="filter_gender")
        if selected_gender != 'All':
            filtered_data = filtered_data[filtered_data['gender'] == selected_gender]
        if len(filtered_data) > 0:
            min_pay = float(filtered_data['tdc1'].min())
            max_pay = float(filtered_data['tdc1'].quantile(0.99) if len(filtered_data) > 10 else filtered_data['tdc1'].max())
        else:
            min_pay = float(data['tdc1'].min())
            max_pay = float(data['tdc1'].quantile(0.99))
        pay_range = st.slider("Annual Compensation Range", min_value=min_pay/1000, max_value=max_pay/1000, value=(min_pay/1000, max_pay/1000), step=100.0, format="$%.0fM", key="filter_pay_range")
        pay_range = (pay_range[0]*1000 , pay_range[1]*1000)
    apply_filters = st.sidebar.button("Apply Filters", key="apply_filters_button")
    reset_filters = st.sidebar.button("Reset Filters", key="reset_filters_button")
    filters = {
        'ticker': selected_ticker,
        'industry': selected_industry,
        'year_range': year_range,
        'firm_size': selected_firm_size,
        'title': selected_title,
        'gender': selected_gender,
        'pay_range': pay_range,
        'apply': apply_filters,
        'reset': reset_filters
    }
    return filters

def apply_filters_to_data(data, filters):
    filtered_data = data.copy()
    if filters['ticker'] != 'All Tickers':
        filtered_data = filtered_data[filtered_data['ticker'] == filters['ticker']]
    if filters['industry'] != 'All Industries':
        filtered_data = filtered_data[filtered_data['industry_name'] == filters['industry']]
    filtered_data = filtered_data[(filtered_data['year'] >= filters['year_range'][0]) & (filtered_data['year'] <= filters['year_range'][1])]
    if filters['firm_size'] != 'All Sizes':
        filtered_data = filtered_data[filtered_data['firm_size'] == filters['firm_size']]
    if filters['title'] != 'All Titles':
        filtered_data = filtered_data[filtered_data['title'] == filters['title']]
    if filters['gender'] != 'All':
        filtered_data = filtered_data[filtered_data['gender'] == filters['gender']]
    filtered_data = filtered_data[(filtered_data['tdc1'] >= filters['pay_range'][0]) & (filtered_data['tdc1'] <= filters['pay_range'][1])]
    return filtered_data

# =====================
# SUMMARY STATS & KPI COMPONENT
# =====================
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
            formatted_delta = f"{delta_prefix}{delta:.2f} {delta_suffix}"
    if delta is not None:
        st.metric(label=title, value=formatted_value, delta=formatted_delta, delta_color="normal")
    else:
        st.metric(label=title, value=formatted_value)

def calculate_avg_ceo_pay(data):
    return data['tdc1'].mean()

def get_top_paid_executives(data, n=5):
    return data.sort_values('tdc1', ascending=False).head(n)[['coname', 'title', 'year', 'tdc1', 'gender']]

def calculate_avg_roa_top_paid(data, top_pct=0.25):
    top_paid_threshold = data['tdc1'].quantile(1 - top_pct)
    top_paid_firms = data[data['tdc1'] >= top_paid_threshold]
    return top_paid_firms['roa'].mean()

def calculate_pay_performance_alignment(data):
    pay_roa_corr = data[['tdc1', 'roa']].corr().iloc[0, 1]
    alignment_score = (pay_roa_corr + 1) * 50
    return alignment_score

def create_pay_performance_scorecard(data):
    metrics = {}
    metrics['pay_roa_corr'] = data[['tdc1', 'roa']].corr().iloc[0, 1]
    metrics['pay_ni_corr'] = data[['tdc1', 'ni']].corr().iloc[0, 1]
    metrics['pay_revt_corr'] = data[['tdc1', 'revt']].corr().iloc[0, 1]
    high_pay_threshold = data['tdc1'].quantile(0.75)
    low_perf_threshold = data['roa'].quantile(0.25)
    metrics['pct_high_pay_low_perf'] = len(data[(data['tdc1'] > high_pay_threshold) & (data['roa'] < low_perf_threshold)]) / len(data) * 100
    data['pay_rank'] = data['tdc1'].rank(pct=True)
    data['roa_rank'] = data['roa'].rank(pct=True)
    data['pay_perf_diff'] = (data['pay_rank'] - data['roa_rank']).abs()
    metrics['pct_aligned'] = len(data[data['pay_perf_diff'] < 0.25]) / len(data) * 100
    return metrics

def summary_stats_kpi_component(filtered_data=None):
    st.header("Summary Statistics & Key Performance Indicators")
    if filtered_data is None:
        data = load_data()
        if 'filtered_data' in st.session_state:
            filtered_data = st.session_state['filtered_data']
        else:
            filtered_data = data
    else:
        data = load_data()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_pay = calculate_avg_ceo_pay(filtered_data)
        create_metric_card("Average CEO Pay (in '000)", avg_pay, is_currency=True)
    with col2:
        avg_roa_top = calculate_avg_roa_top_paid(filtered_data)
        avg_roa_all = filtered_data['roa'].mean()
        delta_roa = avg_roa_top - avg_roa_all
        create_metric_card("Avg ROA for Top-Paid Firms", avg_roa_top, delta_roa, "vs. all firms", is_percentage=True)
    with col3:
        alignment_score = calculate_pay_performance_alignment(filtered_data)
        if len(filtered_data) < len(data):
            alignment_score_all = calculate_pay_performance_alignment(data)
            delta_alignment = alignment_score - alignment_score_all
            delta_suffix = "vs. overall alignment"
        else:
            delta_alignment = None
            delta_suffix = ""
        create_metric_card("Pay-Performance Alignment Score", alignment_score, delta_alignment, delta_suffix)
    with col4:
        num_companies = filtered_data['gvkey'].nunique()
        num_executives = filtered_data['execid'].nunique()
        st.metric(label="Number of Companies", value=f"{num_companies:,}", delta=f"{num_executives:,} Executives")
    tab1, tab2 = st.tabs(["Top Paid Executives", "Pay-Performance Scorecard"])
    with tab1:
        top_execs = get_top_paid_executives(filtered_data)
        top_execs_display = top_execs.copy()
        top_execs_display['tdc1'] = top_execs_display['tdc1'].apply(lambda x: f"${x:,.0f}")
        top_execs_display.columns = ['Company', 'Title', 'Year', 'Compensation (in \'000)', 'Gender']
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
        scorecard_metrics = create_pay_performance_scorecard(filtered_data)
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            st.subheader("Pay-Performance Correlations")
            fig = go.Figure()
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
                        {'range': [0, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': scorecard_metrics['pay_roa_corr']
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Pay vs. Net Income Correlation", f"{scorecard_metrics['pay_ni_corr']:.2f}")
            st.metric("Pay vs. Revenue Correlation", f"{scorecard_metrics['pay_revt_corr']:.2f}")
        with st.expander("About these metrics"):
            st.markdown("""
        ### Pay-Performance Scorecard
        - **Pay-Performance Correlations**: The correlation coefficients between CEO pay and various performance metrics.
            - Values range from -1 (perfect negative correlation) to 1 (perfect positive correlation).
            - Values close to 0 indicate little to no correlation.
        - **Pay-Performance Alignment**: The percentage of companies where CEO pay rank (percentile) is within 25 percentage points of their performance rank.
            - If the CEO's pay rank and performance rank are within 25 percentile points of each other (up or down), it is considered "aligned."
            - Example: If a CEO's pay rank is at the 80th percentile and the company's performance is at the 65th percentile → difference is 15 points → considered aligned.
        - **High Pay, Low Performance**: The percentage of companies with high CEO pay (top 25%) but low performance (bottom 25%).""")
        with sc_col2:
            st.subheader("Pay-Performance Alignment")
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=["Aligned", "Misaligned"],
                values=[scorecard_metrics['pct_aligned'], 100 - scorecard_metrics['pct_aligned']],
                hole=0.7,
                marker_colors=['green', 'red']
            ))
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
            st.metric(
                "High Pay, Low Performance",
                f"{scorecard_metrics['pct_high_pay_low_perf']:.1f}%",
            )

# =====================
# TIME SERIES COMPONENT
# =====================
def create_time_series_chart(data, metric='roa'):
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
        temp_data = data[~data['revenue_growth'].isin([np.inf, -np.inf])]
        temp_data = temp_data.dropna(subset=['revenue_growth'])
        yearly_metric = temp_data.groupby('year')['revenue_growth'].mean().reset_index()
        metric_label = 'Revenue Growth (%)'
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=yearly_pay['year'], y=yearly_pay['tdc1'], name="CEO Compensation", line=dict(color='blue', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=yearly_metric['year'], y=yearly_metric[metric], name=metric_label, line=dict(color='red', width=2)), secondary_y=True)
    fig.update_layout(title_text="CEO Pay and Performance Over Time", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=500)
    fig.update_yaxes(title_text="Average CEO Compensation", secondary_y=False)
    fig.update_yaxes(title_text=metric_label, secondary_y=True)
    # Alignment markers logic
    if len(yearly_pay) > 1 and len(yearly_metric) > 1:
        pay_change = yearly_pay['tdc1'].pct_change().dropna()
        metric_change = yearly_metric[metric].pct_change().dropna()
        common_years = set(pay_change.index).intersection(set(metric_change.index))
        pay_change = pay_change.loc[list(common_years)]
        metric_change = metric_change.loc[list(common_years)]
        alignment = (pay_change * metric_change > 0).astype(int)
        alignment_years = yearly_pay['year'].iloc[list(common_years)]
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
                        showlegend=i==len(alignment)-1
                    ),
                    secondary_y=False,
                )
    return fig

def time_series_component(filtered_data=None):
    st.header("Time Series Analysis")
    if filtered_data is None:
        data = load_data()
        if 'filtered_data' in st.session_state:
            filtered_data = st.session_state['filtered_data']
        else:
            filtered_data = data
    if len(filtered_data) == 0:
        st.warning("No data available for time series analysis with current filters.")
        return
    if filtered_data['year'].nunique() < 2:
        st.warning("Time series analysis requires data from at least two years. Please adjust your filters.")
        return
    metric_options = {'roa': 'Return on Assets (ROA)', 'ni': 'Net Income', 'revt': 'Revenue', 'revenue_growth': 'Revenue Growth'}
    if 'filters' in st.session_state and 'performance_metric' in st.session_state['filters']:
        default_metric = st.session_state['filters']['performance_metric']
    else:
        default_metric = 'roa'
    selected_metric = default_metric
    fig = create_time_series_chart(filtered_data, selected_metric)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("About this visualization"):
        st.markdown("""
        This chart shows the relationship between average CEO compensation and company performance over time.
        - **Blue line**: Average CEO compensation (tdc1) for each year
        - **Red line**: Average performance metric for each year
        - Alignment occurs when both CEO pay and the performance metric move in the same direction year-over-year.
        """)

# =====================
# SCATTER PLOT COMPONENT
# =====================
def create_scatter_plot(data, x_col='tdc1', y_col='roa', color_col=None):
    plot_data = data.dropna(subset=[x_col, y_col])
    if color_col:
        fig = px.scatter(plot_data, x=x_col, y=y_col, color=color_col, opacity=0.7, trendline="ols", trendline_color_override="red", hover_data=['coname', 'year', 'tdc1', 'roa', 'ni', 'revt'])
    else:
        fig = px.scatter(plot_data, x=x_col, y=y_col, opacity=0.7, trendline="ols", trendline_color_override="red", hover_data=['coname', 'year', 'tdc1', 'roa', 'ni', 'revt'])
    fig.update_layout(title=f"CEO Compensation vs {y_col.upper() if y_col in ['roa', 'ni'] else y_col.title()}", xaxis_title="CEO Compensation (tdc1)", yaxis_title=y_col.upper() if y_col in ['roa', 'ni'] else y_col.title(), height=600)
    high_pay_threshold = plot_data[x_col].quantile(0.75)
    low_perf_threshold = plot_data[y_col].quantile(0.25)
    outliers = plot_data[(plot_data[x_col] > high_pay_threshold) & (plot_data[y_col] < low_perf_threshold)]
    if not outliers.empty:
        fig.add_trace(go.Scatter(x=outliers[x_col], y=outliers[y_col], mode='markers', marker=dict(color='red', size=12, line=dict(color='black', width=2)), name="Pay-for-No-Performance Outliers", text=outliers['coname'] + " (" + outliers['year'].astype(str) + ")", hoverinfo="text"))
    X = plot_data[x_col].values.reshape(-1, 1)
    y = plot_data[y_col].values
    model = LinearRegression()
    model.fit(X, y)
    r_squared = model.score(X, y)
    n = len(X)
    p = 1
    t_stat = model.coef_[0] / (np.sqrt((1 - r_squared) / (n - p - 1)) / np.sqrt(np.sum((X - np.mean(X))**2)))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - p - 1))
    fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.98, xanchor='right', text=f"R² = {r_squared:.3f}<br>p-value = {p_value:.3e}<br>Linear Regression Slope = {model.coef_[0]:.3f}", showarrow=False, align='left', font=dict(size=15, color="black"), bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="black", borderwidth=1, borderpad=4)
    return fig, outliers

def scatter_plot_component(filtered_data=None):
    st.header("Scatter Plot Analysis")
    if filtered_data is None:
        data = load_data()
        if 'filtered_data' in st.session_state:
            filtered_data = st.session_state['filtered_data']
        else:
            filtered_data = data
    if len(filtered_data) == 0:
        st.warning("No data available for scatter plot analysis with current filters.")
        return
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_options = {'roa': 'Return on Assets (ROA) %', 'ni': 'Net Income ($)', 'revt': 'Revenue ($)', 'revenue_growth': 'Revenue Growth (%)'}
        if 'filters' in st.session_state and 'performance_metric' in st.session_state['filters']:
            default_metric = st.session_state['filters']['performance_metric']
        else:
            default_metric = 'roa'
        selected_metric = st.selectbox("Select Performance Metric (Y-axis)", options=list(metric_options.keys()), format_func=lambda x: metric_options[x], index=list(metric_options.keys()).index(default_metric), key="scatter_metric")
    with col2:
        color_options = {None: 'None', 'industry_name': 'Industry', 'firm_size': 'Firm Size', 'gender': 'Gender', 'year': 'Year'}
        selected_color = st.selectbox("Color Points By", options=list(color_options.keys()), format_func=lambda x: color_options[x] if x is not None else 'None', key="scatter_color")
    with col3:
        if len(filtered_data) > 0:
            min_assets = st.slider("Minimum Firm Size (Assets in $)", min_value=float(filtered_data['at'].min()), max_value=float(filtered_data['at'].max()), value=float(filtered_data['at'].min()), format="$%.0f", key="scatter_min_assets")
        else:
            min_assets = 0
    scatter_filtered_data = filtered_data[filtered_data['at'] >= min_assets]
    if len(scatter_filtered_data) == 0:
        st.warning("No data available after applying minimum assets filter. Please adjust your filters.")
        return
    fig, outliers = create_scatter_plot(scatter_filtered_data, 'tdc1', selected_metric, selected_color)
    st.plotly_chart(fig, use_container_width=True)
    if not outliers.empty:
        with st.expander("Pay-for-No-Performance Outliers"):
            st.write(f"Found {len(outliers)} companies with high pay but low {metric_options[selected_metric]}:")
            st.dataframe(outliers[['coname', 'year', 'tdc1', selected_metric, 'industry_name']].sort_values('tdc1', ascending=False), hide_index=True)
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

# =====================
# CORRELATION HEATMAP COMPONENT
# =====================
def create_correlation_heatmap(data, variables=None, industry=None):
    if industry and industry != 'All Industries':
        filtered_data = data[data['industry_name'] == industry].copy()
    else:
        filtered_data = data.copy()
    if variables is None:
        variables = ['tdc1', 'roa', 'ni', 'revt', 'at', 'mkvalt', 'tenure']
    var_mapping = {'tdc1': 'CEO Pay', 'roa': 'ROA (%)', 'ni': 'Net Income', 'revt': 'Revenue', 'at': 'Total Assets', 'mkvalt': 'Market Value', 'tenure': 'CEO Tenure', 'emp': 'Employees', 'age': 'CEO Age', 'revenue_growth': 'Rev Growth (%)'}
    numeric_data = filtered_data[variables].copy()
    corr_matrix = numeric_data.corr()
    corr_matrix.columns = [var_mapping.get(col, col) for col in corr_matrix.columns]
    corr_matrix.index = [var_mapping.get(idx, idx) for idx in corr_matrix.index]
    fig = px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(title=f"Correlation Heatmap {f'for {industry}' if industry and industry != 'All Industries' else ''}", height=600, coloraxis_colorbar=dict(title="Correlation", thicknessmode="pixels", thickness=20, lenmode="pixels", len=300, tickvals=[-1, -0.5, 0, 0.5, 1], ticktext=['-1<br>(Perfect<br>Negative)', '-0.5', '0<br>(No<br>Correlation)', '0.5', '1<br>(Perfect<br>Positive)']))
    fig.update_traces(hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>')
    return fig, corr_matrix

def create_sector_comparison(data, variable='tdc1'):
    # Get top industries by count
    top_industries = data['industry_name'].value_counts().head(8).index.tolist()
    # Filter data to include only top industries
    filtered_data = data[data['industry_name'].isin(top_industries)].copy()
    # Create a pivot table of correlations between the selected variable and ROA for each industry
    industry_correlations = {}
    for industry in top_industries:
        industry_data = filtered_data[filtered_data['industry_name'] == industry]
        if len(industry_data) > 5:  # Ensure enough data points
            corr = industry_data[['tdc1', 'roa', 'ni', 'revt', 'at', 'mkvalt']].corr()
            industry_correlations[industry] = corr[variable].drop(variable)
    # Convert to DataFrame
    comparison_df = pd.DataFrame(industry_correlations)
    # Create a more readable mapping for variable names
    var_mapping = {
        'roa': 'ROA (%)',
        'ni': 'Net Income',
        'revt': 'Revenue',
        'at': 'Total Assets',
        'mkvalt': 'Market Value'
    }
    # Rename index for better readability
    comparison_df.index = [var_mapping.get(idx, idx) for idx in comparison_df.index]
    # Create heatmap using plotly
    fig = px.imshow(
        comparison_df,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        aspect="auto"
    )
    # Update layout
    fig.update_layout(
        title=f"Correlation of {variable.upper() if variable in ['roa', 'ni'] else variable.title()} with Performance Metrics Across Industries",
        height=500,
        xaxis_title="Industry",
        yaxis_title="Performance Metric",
        coloraxis_colorbar=dict(
            title="Correlation",
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=300,
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['-1<br>(Perfect<br>Negative)', '-0.5', '0<br>(No<br>Correlation)', '0.5', '1<br>(Perfect<br>Positive)']
        ),
        xaxis_tickangle=-45
    )
    # Add hover template
    fig.update_traces(
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    )
    return fig, comparison_df

def correlation_heatmap_component(filtered_data=None):
    st.header("Correlation Analysis")
    if filtered_data is None:
        data = load_data()
        if 'filtered_data' in st.session_state:
            filtered_data = st.session_state['filtered_data']
        else:
            filtered_data = data
    if len(filtered_data) == 0:
        st.warning("No data available for correlation analysis with current filters.")
        return
    tab1, tab2 = st.tabs(["Variable Correlation Heatmap", "Industry Comparison"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            available_industries = ['All Industries'] + sorted(filtered_data['industry_name'].unique().tolist())
            selected_industry = st.selectbox("Select Industry", options=available_industries, key="heatmap_industry")
        with col2:
            all_variables = ['tdc1', 'roa', 'ni', 'revt', 'at', 'mkvalt', 'tenure', 'emp', 'age', 'revenue_growth']
            var_mapping = {'tdc1': 'CEO Pay', 'roa': 'ROA (%)', 'ni': 'Net Income', 'revt': 'Revenue', 'at': 'Total Assets', 'mkvalt': 'Market Value', 'tenure': 'CEO Tenure', 'emp': 'Employees', 'age': 'CEO Age', 'revenue_growth': 'Rev Growth (%)'}
            available_variables = [var for var in all_variables if var in filtered_data.columns]
            default_vars = ['tdc1', 'roa', 'ni', 'revt', 'at', 'mkvalt', 'tenure']
            default_vars = [var for var in default_vars if var in available_variables]
            selected_variables = st.multiselect("Select Variables", options=available_variables, default=default_vars, format_func=lambda x: var_mapping.get(x, x), key="heatmap_variables")
            if not selected_variables:
                selected_variables = default_vars
        try:
            fig, corr_matrix = create_correlation_heatmap(filtered_data, selected_variables, selected_industry)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("About this visualization"):
                st.markdown("""
            This heatmap shows the correlation between different variables in the dataset.

            - Correlation ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation)
            - Values close to 0 indicate little to no correlation
            - Blue colors indicate negative correlation (one variable increases as the other decreases)
            - Red colors indicate positive correlation (variables increase or decrease together)

            You can filter by industry and select which variables to include in the correlation matrix.""")
        except Exception as e:
            st.warning(f"Could not create heatmap: {e}")
    with tab2:
        # Industry comparison heatmap
        if filtered_data['industry_name'].nunique() < 2:
            st.warning("Industry comparison requires data from at least two industries. Please adjust your filters.")
        else:
            selected_var = 'tdc1'  # Could be made a selectbox if you want to allow user to choose
            try:
                fig, comp_df = create_sector_comparison(filtered_data, selected_var)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating industry comparison: {str(e)}. Try adjusting filters to include more industries.")
            with st.expander("About this visualization"):
                st.markdown("""
                This heatmap compares how the selected variable correlates with performance metrics across different industries.
                - Each column represents an industry
                - Each row represents a performance metric
                - The color indicates the strength and direction of correlation
                This visualization helps identify which industries show stronger relationships between the selected variable and performance metrics.
                """)

# =====================
# BAR CHARTS & BOX PLOTS COMPONENT (Simplified)
# =====================
def bar_charts_box_plots_component(filtered_data):
    st.header("Comparative Analysis")
    if len(filtered_data) == 0:
        st.warning("No data available for comparative analysis with current filters.")
        return
    tab1, tab3 = st.tabs(["Compensation by Category", "Compensation Trends"])
    color_scale = "Viridis"
    line_color = "#00CED1"
    template = "plotly_dark"
    with tab1:
        st.subheader("Average CEO Compensation by Industry")
        grouped_data = filtered_data.groupby('industry_name')['tdc1'].mean().reset_index()
        grouped_data.columns = ['industry_name', 'avg_compensation']
        if len(grouped_data) == 0:
            st.warning("No industry data available with current filters.")
            return
        grouped_data = grouped_data.sort_values('avg_compensation', ascending=False).head(10)
        fig = px.bar(grouped_data, x='industry_name', y='avg_compensation', labels={'industry_name': 'Industry', 'avg_compensation': 'Average CEO Compensation'}, title='Average CEO Compensation by Industry', color='avg_compensation', color_continuous_scale=color_scale, template=template)
        fig.update_yaxes(tickprefix='$', tickformat=',')
        fig.update_layout(xaxis_tickangle=-45)
        fig.update_layout(title_font=dict(size=20, color="white"), legend_title_font=dict(size=14, color="white"), legend_font=dict(size=12, color="white"), xaxis_title_font=dict(size=14, color="white"), yaxis_title_font=dict(size=14, color="white"), coloraxis_showscale=False, margin=dict(t=50, b=50, l=50, r=50), height=500)
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        st.subheader("CEO Compensation Trends Over Time")
        if filtered_data['year'].nunique() < 2:
            st.warning("Insufficient data for trend analysis. Trend analysis requires data from multiple years.")
        else:
            trend_data = filtered_data.groupby('year')['tdc1'].mean().reset_index()
            fig = px.line(trend_data, x='year', y='tdc1', labels={'year': 'Year', 'tdc1': 'Average Annual CEO Compensation'}, title='Average CEO Compensation Over Time', template=template)
            fig.update_yaxes(tickprefix='$', tickformat=',')
            fig.update_traces(line=dict(color=line_color, width=3), marker=dict(size=10, color=line_color))
            fig.update_layout(title_font=dict(size=20, color="white"), legend_title_font=dict(size=14, color="white"), legend_font=dict(size=12, color="white"), xaxis_title_font=dict(size=14, color="white"), yaxis_title_font=dict(size=14, color="white"), margin=dict(t=50, b=50, l=50, r=50), height=500)
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("CEO Compensation vs ROA")
        if 'roa' not in filtered_data.columns or filtered_data['roa'].isna().all():
            st.warning("ROA data not available with current filters.")
        else:
            try:
                filtered_data['roa_quartile'] = pd.qcut(filtered_data['roa'], 4, labels=['Bottom 25%', 'Lower Middle', 'Upper Middle', 'Top 25%'])
                performance_data = filtered_data.groupby('roa_quartile')['tdc1'].mean().reset_index()
                fig = px.bar(performance_data, x='roa_quartile', y='tdc1', labels={'roa_quartile': 'ROA Performance Quartile', 'tdc1': 'Average Annual CEO Compensation'}, title='Average CEO Compensation by ROA Performance', category_orders={'roa_quartile': ['Bottom 25%', 'Lower Middle', 'Upper Middle', 'Top 25%']}, color='tdc1', color_continuous_scale=color_scale, template=template)
                fig.update_yaxes(tickprefix='$', tickformat=',')
                fig.update_layout(title_font=dict(size=20, color="white"), legend_title_font=dict(size=14, color="white"), legend_font=dict(size=12, color="white"), xaxis_title_font=dict(size=14, color="white"), yaxis_title_font=dict(size=14, color="white"), coloraxis_showscale=False, margin=dict(t=50, b=50, l=50, r=50), height=500)
                st.plotly_chart(fig, use_container_width=True)
            except ValueError as e:
                st.warning("Insufficient unique ROA values to create performance quartiles. Try adjusting filters to include more diverse data.")
                st.subheader("CEO Compensation vs ROA (Scatter Plot)")
                fig = px.scatter(filtered_data, x='roa', y='tdc1', labels={'roa': 'Return on Assets (ROA) %', 'tdc1': 'Annual CEO Compensation'})
                st.plotly_chart(fig, use_container_width=True)

# =====================
# MAIN DASHBOARD LOGIC (from app.py)
# =====================
st.set_page_config(
    page_title="CEO Pay and Performance Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .main .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    h1, h2, h3 {margin-top: 0.5rem; margin-bottom: 0.5rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 2px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; padding-bottom: 10px;}
    .stTabs [aria-selected="true"] {border-bottom: 2px solid var(--primary-color);}
    div.stMetric {border-radius: 5px; padding: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.1);}
    .metric-row {display: flex; justify-content: space-between;}
    .stExpander {border-radius: 5px;}
    .stPlotlyChart {margin-top: 1rem; margin-bottom: 1rem; border-radius: 5px; overflow: hidden;}
    .sidebar .sidebar-content {padding-top: 1rem;}
    .stSlider {padding-top: 1rem; padding-bottom: 1rem;}
    .stSelectbox {margin-bottom: 0.5rem;}
    .stTable {border-radius: 5px; overflow: hidden;}
    .stAlert {border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

def main():
    st.title("CEO Pay and Performance Dashboard")
    st.markdown("Contributors: Aastha Surana, Kyle Chin, Matthew Maslow")
    st.markdown("Exploring the relationship between executive compensation and firm performance")
    if 'filtered_data' not in st.session_state:
        data = load_data()
        st.session_state['filtered_data'] = data
        st.session_state['filters'] = {}
    filters = create_filters_sidebar()
    data = load_data()
    if filters['apply'] or filters['ticker'] != 'All Tickers':
        filtered_data = apply_filters_to_data(data, filters)
        st.session_state['filtered_data'] = filtered_data
        st.session_state['filters'] = filters
    elif filters['reset']:
        st.session_state['filtered_data'] = data
        st.session_state['filters'] = filters
    filtered_data = st.session_state['filtered_data']
    if len(filtered_data) == 0:
        st.warning("No data matches the selected filters. Please adjust your filter criteria.")
        return
    summary_stats_kpi_component(filtered_data)
    tab1,tab2, tab3, tab4 = st.tabs(["Time Series Analysis", "Scatter Plot Analysis", "Correlation Analysis", "Comparative Analysis"])
    with tab1:
        time_series_component(filtered_data)
    with tab2:
        scatter_plot_component(filtered_data)
    with tab3:
        correlation_heatmap_component(filtered_data)
    with tab4:
        bar_charts_box_plots_component(filtered_data)
    st.markdown("---")
    st.markdown("""
    **Data Source:** WRDS (ExecuComp + Compustat) for U.S. companies from 2010-2024  
    **Last Updated:** April 2025  
    """)

if __name__ == "__main__":
    main() 