import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

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

# Function to create correlation heatmap
def create_correlation_heatmap(data, variables=None, industry=None):
    # Filter data by industry if specified
    if industry and industry != 'All Industries':
        filtered_data = data[data['industry_name'] == industry].copy()
    else:
        filtered_data = data.copy()
    
    # Default variables if none specified
    if variables is None:
        variables = ['tdc1', 'roa', 'ni', 'revt', 'at', 'mkvalt', 'tenure']
    
    # Create a more readable mapping for variable names
    var_mapping = {
        'tdc1': 'CEO Pay',
        'roa': 'ROA (%)',
        'ni': 'Net Income',
        'revt': 'Revenue',
        'at': 'Total Assets',
        'mkvalt': 'Market Value',
        'tenure': 'CEO Tenure',
        'emp': 'Employees',
        'age': 'CEO Age',
        'revenue_growth': 'Rev Growth (%)'
    }
    
    # Select only numeric columns for correlation
    numeric_data = filtered_data[variables].copy()
    
    # Calculate correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Rename columns and index for better readability
    corr_matrix.columns = [var_mapping.get(col, col) for col in corr_matrix.columns]
    corr_matrix.index = [var_mapping.get(idx, idx) for idx in corr_matrix.index]
    
    # Create heatmap using plotly
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        aspect="auto"
    )
    
    # Update layout
    fig.update_layout(
        title=f"Correlation Heatmap {f'for {industry}' if industry and industry != 'All Industries' else ''}",
        height=600,
        coloraxis_colorbar=dict(
            title="Correlation",
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=300,
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['-1<br>(Perfect<br>Negative)', '-0.5', '0<br>(No<br>Correlation)', '0.5', '1<br>(Perfect<br>Positive)']
        )
    )
    
    # Add hover template
    fig.update_traces(
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    )
    
    return fig, corr_matrix

# Function to create sector comparison heatmap
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

# Main function to run the correlation heatmap component
def correlation_heatmap_component(filtered_data=None):
    st.header("Correlation Analysis")
    
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
        st.warning("No data available for correlation analysis with current filters.")
        return
    
    # Create tabs for different correlation views
    tab1, tab2 = st.tabs(["Variable Correlation Heatmap", "Industry Comparison"])
    
    with tab1:
        # Create columns for controls
        col1, col2 = st.columns(2)
        
        with col1:
            # Create industry selector - dynamically updated based on filtered data
            available_industries = ['All Industries'] + sorted(filtered_data['industry_name'].unique().tolist())
            selected_industry = st.selectbox(
                "Select Industry",
                options=available_industries,
                key="heatmap_industry"
            )
        
        with col2:
            # Create variable selector
            all_variables = ['tdc1', 'roa', 'ni', 'revt', 'at', 'mkvalt', 'tenure', 'emp', 'age', 'revenue_growth']
            var_mapping = {
                'tdc1': 'CEO Pay',
                'roa': 'ROA (%)',
                'ni': 'Net Income',
                'revt': 'Revenue',
                'at': 'Total Assets',
                'mkvalt': 'Market Value',
                'tenure': 'CEO Tenure',
                'emp': 'Employees',
                'age': 'CEO Age',
                'revenue_growth': 'Rev Growth (%)'
            }
            
            # Filter available variables based on what's in the filtered data
            available_variables = [var for var in all_variables if var in filtered_data.columns]
            
            default_vars = ['tdc1', 'roa', 'ni', 'revt', 'at', 'mkvalt', 'tenure']
            # Ensure default vars are in available variables
            default_vars = [var for var in default_vars if var in available_variables]
            
            selected_variables = st.multiselect(
                "Select Variables",
                options=available_variables,
                default=default_vars,
                format_func=lambda x: var_mapping.get(x, x),
                key="heatmap_variables"
            )
            
            if not selected_variables:
                selected_variables = default_vars
        
        # Create and display the correlation heatmap
        try:
            fig, corr_matrix = create_correlation_heatmap(filtered_data, selected_variables, selected_industry)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {str(e)}. Try selecting different variables or adjusting filters.")
        
        # Add explanation
        with st.expander("About this visualization"):
            st.markdown("""
            This heatmap shows the correlation between different variables in the dataset.
            
            - Correlation ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation)
            - Values close to 0 indicate little to no correlation
            - Blue colors indicate negative correlation (one variable increases as the other decreases)
            - Red colors indicate positive correlation (variables increase or decrease together)
            
            You can filter by industry and select which variables to include in the correlation matrix.
            """)
    
    with tab2:
        # Check if we have enough industries for comparison
        if filtered_data['industry_name'].nunique() < 2:
            st.warning("Industry comparison requires data from at least two industries. Please adjust your filters.")
        else:
            # Create variable selector for industry comparison
            # selected_var = st.selectbox(
            #     "Select Variable to Compare Across Industries",
            #     #options=['tdc1', 'roa', 'ni', 'revt', 'at', 'mkvalt'],
            #     options=['tdc1'],
            #     format_func=lambda x: var_mapping.get(x, x),
            #     key="industry_comp_var"
            # )
            selected_var = 'tdc1'
            
            # Create and display the sector comparison heatmap
            try:
                fig, comp_df = create_sector_comparison(filtered_data, selected_var)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating industry comparison: {str(e)}. Try adjusting filters to include more industries.")
            
            # Add explanation
            with st.expander("About this visualization"):
                st.markdown("""
                This heatmap compares how the selected variable correlates with performance metrics across different industries.
                
                - Each column represents an industry
                - Each row represents a performance metric
                - The color indicates the strength and direction of correlation
                
                This visualization helps identify which industries show stronger relationships between the selected variable and performance metrics.
                """)

if __name__ == "__main__":
    correlation_heatmap_component()
