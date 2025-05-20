import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    
    # Calculate performance quartiles
    data['roa_quartile'] = pd.qcut(data['roa'], 4, labels=['Bottom 25%', 'Lower Middle', 'Upper Middle', 'Top 25%'])
    
    return data

# Function to create bar chart for average compensation by category
def create_compensation_bar_chart(data, category='industry_name', top_n=10):
    # Group by category and calculate average compensation
    grouped_data = data.groupby(category)['tdc1'].agg(['mean', 'count']).reset_index()
    grouped_data.columns = [category, 'avg_compensation', 'count']
    
    # Check if we have any data after filtering
    if len(grouped_data) == 0:
        return None
    
    # For test data or small datasets, don't filter by count
    if len(data) <= 10:
        min_count = 1
    else:
        min_count = 2  # Reduced from 5 to 2 to be more lenient
    
    # Filter to include only categories with sufficient data points
    filtered_data = grouped_data[grouped_data['count'] >= min_count]
    
    # If filtering removed all data, use original grouped data
    if len(filtered_data) == 0:
        filtered_data = grouped_data
    
    # Sort by average compensation and get top N
    filtered_data = filtered_data.sort_values('avg_compensation', ascending=False).head(top_n)
    
    # Create bar chart
    fig = px.bar(
        filtered_data,
        x=category,
        y='avg_compensation',
        text='count',
        labels={
            category: category.replace('_', ' ').title(),
            'avg_compensation': 'Average CEO Compensation',
            'count': 'Number of Records'
        },
        title=f'Average CEO Compensation by {category.replace("_", " ").title()}',
        color='avg_compensation',
        color_continuous_scale='Viridis'
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        xaxis_title=category.replace('_', ' ').title(),
        yaxis_title='Average CEO Compensation',
        coloraxis_showscale=False
    )
    
    # Format y-axis as currency
    fig.update_yaxes(tickprefix='$', tickformat=',')
    
    # Add data count as text labels
    fig.update_traces(
        texttemplate='%{text} records',
        textposition='outside'
    )
    
    return fig

# Function to create box plot for compensation vs performance groups
def create_compensation_box_plot(data, group_by='roa_quartile'):
    # Create a copy of the data to avoid modifying the original
    plot_data = data.copy()
    
    # Ensure roa_quartile exists if that's the selected group
    if group_by == 'roa_quartile' and 'roa_quartile' not in plot_data.columns:
        try:
            # Create performance quartiles
            plot_data['roa_quartile'] = pd.qcut(plot_data['roa'], 4, 
                                          labels=['Bottom 25%', 'Lower Middle', 'Upper Middle', 'Top 25%'])
        except ValueError:
            # Handle case where there are not enough distinct values for quartiles
            return None
    
    # Check if we have enough data for the selected group
    if plot_data[group_by].nunique() < 2:
        return None
    
    # Create box plot
    fig = px.box(
        plot_data,
        x=group_by,
        y='tdc1',
        color=group_by,
        labels={
            group_by: group_by.replace('_', ' ').title(),
            'tdc1': 'CEO Compensation'
        },
        title=f'CEO Compensation Distribution by {group_by.replace("_", " ").title()}',
        category_orders={
            'roa_quartile': ['Bottom 25%', 'Lower Middle', 'Upper Middle', 'Top 25%'],
            'firm_size': ['Small', 'Medium', 'Large', 'Very Large']
        }
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        xaxis_title=group_by.replace('_', ' ').title(),
        yaxis_title='CEO Compensation',
        showlegend=False
    )
    
    # Format y-axis as currency
    fig.update_yaxes(tickprefix='$', tickformat=',')
    
    return fig

# Function to create compensation trend by year and category
def create_compensation_trend_chart(data, category='industry_name', top_n=5):
    # Check if we have enough years for a trend
    if data['year'].nunique() < 2:
        return None
    
    # Get top categories by count
    top_categories = data[category].value_counts().head(top_n).index.tolist()
    
    # Filter data to include only top categories
    filtered_data = data[data[category].isin(top_categories)]
    
    # Group by year and category
    grouped_data = filtered_data.groupby(['year', category])['tdc1'].mean().reset_index()
    
    # Create line chart
    fig = px.line(
        grouped_data,
        x='year',
        y='tdc1',
        color=category,
        labels={
            'year': 'Year',
            'tdc1': 'Average CEO Compensation',
            category: category.replace('_', ' ').title()
        },
        title=f'CEO Compensation Trends by {category.replace("_", " ").title()} (Top {top_n})',
        markers=True
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        xaxis_title='Year',
        yaxis_title='Average CEO Compensation',
        legend_title=category.replace('_', ' ').title()
    )
    
    # Format y-axis as currency
    fig.update_yaxes(tickprefix='$', tickformat=',')
    
    return fig

# Function to create compensation vs performance comparison
def create_compensation_performance_chart(data, performance_metric='roa'):
    # Create a copy of the data to avoid modifying the original
    plot_data = data.copy()
    
    # Create performance quartiles
    if performance_metric not in plot_data.columns:
        return None
    
    # Create quartiles for the selected metric
    metric_quartile = f'{performance_metric}_quartile'
    if metric_quartile not in plot_data.columns:
        try:
            plot_data[metric_quartile] = pd.qcut(plot_data[performance_metric], 4, 
                                            labels=['Bottom 25%', 'Lower Middle', 'Upper Middle', 'Top 25%'])
        except ValueError:
            # Handle case where there are not enough distinct values for quartiles
            return None
    
    # Group by performance quartile and calculate average compensation
    grouped_data = plot_data.groupby(metric_quartile)['tdc1'].mean().reset_index()
    
    # Create bar chart
    fig = px.bar(
        grouped_data,
        x=metric_quartile,
        y='tdc1',
        color='tdc1',
        labels={
            metric_quartile: f'{performance_metric.upper() if performance_metric in ["roa", "ni"] else performance_metric.title()} Quartile',
            'tdc1': 'Average CEO Compensation'
        },
        title=f'CEO Compensation by {performance_metric.upper() if performance_metric in ["roa", "ni"] else performance_metric.title()} Performance',
        color_continuous_scale='Viridis',
        category_orders={
            metric_quartile: ['Bottom 25%', 'Lower Middle', 'Upper Middle', 'Top 25%']
        }
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        xaxis_title=f'{performance_metric.upper() if performance_metric in ["roa", "ni"] else performance_metric.title()} Quartile',
        yaxis_title='Average CEO Compensation',
        coloraxis_showscale=False
    )
    
    # Format y-axis as currency
    fig.update_yaxes(tickprefix='$', tickformat=',')
    
    return fig

# Main function to run the bar charts and box plots component
def bar_charts_box_plots_component(filtered_data=None):
    st.header("Comparative Analysis")
    
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
        st.warning("No data available for comparative analysis with current filters.")
        return
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Compensation by Category", 
        "Compensation Distribution", 
        "Compensation Trends",
        "Pay vs Performance"
    ])
    
    with tab1:
        # Create category selector
        categories = {
            'industry_name': 'Industry',
            'firm_size': 'Firm Size',
            'gender': 'Gender',
            'year': 'Year'
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_category = st.selectbox(
                "Group By",
                options=list(categories.keys()),
                format_func=lambda x: categories[x],
                key="bar_chart_category"
            )
        
        with col2:
            # Adjust max value based on number of unique values in the selected category
            max_categories = min(15, filtered_data[selected_category].nunique())
            
            # Handle case where there's only one category or fewer than 3 categories
            if max_categories < 3:
                st.info(f"Only {max_categories} {selected_category.replace('_', ' ')} categories available with current filters.")
                top_n = max_categories
            else:
                top_n = st.slider(
                    "Number of Categories to Show",
                    min_value=1,
                    max_value=max_categories,
                    value=min(10, max_categories),
                    step=1,
                    key="bar_chart_top_n"
                )
        
        # Create and display the bar chart
        fig = create_compensation_bar_chart(filtered_data, selected_category, top_n)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Insufficient data for bar chart. Try selecting a different category or adjusting filters.")
        
        # Add explanation
        with st.expander("About this visualization"):
            st.markdown("""
            This bar chart shows the average CEO compensation across different categories.
            
            - Each bar represents a category (e.g., industry, firm size)
            - The height of the bar indicates the average compensation
            - The number above each bar shows how many records are in that category
            
            You can change the grouping category and the number of categories to display.
            """)
    
    with tab2:
        # Create group selector for box plot
        groups = {
            'roa_quartile': 'ROA Performance Quartile',
            'firm_size': 'Firm Size',
            'industry_name': 'Industry',
            'gender': 'Gender'
        }
        
        selected_group = st.selectbox(
            "Group By",
            options=list(groups.keys()),
            format_func=lambda x: groups[x],
            key="box_plot_group"
        )
        
        # Create and display the box plot
        fig = create_compensation_box_plot(filtered_data, selected_group)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Insufficient data for box plot with {groups[selected_group]}. Try selecting a different grouping or adjusting filters.")
        
        # Add explanation
        with st.expander("About this visualization"):
            st.markdown("""
            This box plot shows the distribution of CEO compensation across different groups.
            
            - The box represents the middle 50% of compensation values (interquartile range)
            - The line inside the box is the median compensation
            - The whiskers extend to the minimum and maximum values (excluding outliers)
            - Points beyond the whiskers are outliers
            
            This visualization helps identify differences in compensation distribution across groups and detect potential outliers.
            """)
    
    with tab3:
        # Create category selector for trend chart
        trend_categories = {
            'industry_name': 'Industry',
            'firm_size': 'Firm Size',
            'gender': 'Gender'
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_trend_category = st.selectbox(
                "Group By",
                options=list(trend_categories.keys()),
                format_func=lambda x: trend_categories[x],
                key="trend_chart_category"
            )
        
        with col2:
            # Adjust max value based on number of unique values in the selected category
            max_trend_categories = min(10, filtered_data[selected_trend_category].nunique())
            
            # Handle case where there's only one category or fewer than 2 categories
            if max_trend_categories < 2:
                st.info(f"Only {max_trend_categories} {selected_trend_category.replace('_', ' ')} categories available with current filters.")
                top_n_trend = max_trend_categories
            else:
                top_n_trend = st.slider(
                    "Number of Categories to Show",
                    min_value=1,
                    max_value=max_trend_categories,
                    value=min(5, max_trend_categories),
                    step=1,
                    key="trend_chart_top_n"
                )
        
        # Create and display the trend chart
        fig = create_compensation_trend_chart(filtered_data, selected_trend_category, top_n_trend)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient data for trend chart. Trend analysis requires data from multiple years.")
        
        # Add explanation
        with st.expander("About this visualization"):
            st.markdown("""
            This line chart shows how CEO compensation has changed over time for different categories.
            
            - Each line represents a category (e.g., industry, firm size)
            - The y-axis shows the average CEO compensation
            - The x-axis represents years
            
            This visualization helps identify trends and patterns in CEO compensation over time across different categories.
            """)
    
    with tab4:
        # Create performance metric selector
        performance_metrics = {
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
        
        selected_performance = st.selectbox(
            "Performance Metric",
            options=list(performance_metrics.keys()),
            format_func=lambda x: performance_metrics[x],
            index=list(performance_metrics.keys()).index(default_metric),
            key="performance_chart_metric"
        )
        
        # Create and display the performance comparison chart
        fig = create_compensation_performance_chart(filtered_data, selected_performance)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Insufficient data for {performance_metrics[selected_performance]} quartile analysis. Try selecting a different performance metric or adjusting filters.")
        
        # Add explanation
        with st.expander("About this visualization"):
            st.markdown("""
            This bar chart shows the relationship between CEO compensation and company performance.
            
            - Companies are divided into quartiles based on the selected performance metric
            - Each bar represents the average CEO compensation for that performance quartile
            - The chart helps visualize whether higher-performing companies pay their CEOs more
            
            You can change the performance metric to explore different aspects of company performance.
            """)

if __name__ == "__main__":
    bar_charts_box_plots_component()
