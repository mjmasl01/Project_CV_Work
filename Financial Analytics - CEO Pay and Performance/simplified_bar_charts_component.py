import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def bar_charts_box_plots_component(filtered_data):
    """
    Create bar charts and box plots for CEO compensation analysis.
    
    Parameters:
    -----------
    filtered_data : pandas.DataFrame
        Filtered dataset containing CEO compensation and company performance metrics
    """
    st.header("Comparative Analysis")
    
    # Check if filtered data is empty
    if len(filtered_data) == 0:
        st.warning("No data available for comparative analysis with current filters.")
        return
    
    # Create tabs for different visualizations
    tab1, tab3 = st.tabs([
        "Compensation by Category", 
        #"Compensation Distribution", 
        "Compensation Trends",
        #"Compensation vs Performance"
    ])
    
    # Define color schemes that work well in both light and dark modes
    color_scale = "Viridis"  # Works well in both light and dark modes
    line_color = "#00CED1"   # Bright teal that's visible in both modes
    
    # Template that works well in dark mode
    template = "plotly_dark"
    
    # Simple bar chart directly in the main component
    with tab1:
        st.subheader("Average CEO Compensation by Industry")
        
        # Group by industry and calculate average compensation
        grouped_data = filtered_data.groupby('industry_name')['tdc1'].mean().reset_index()
        grouped_data.columns = ['industry_name', 'avg_compensation']
        
        # Check if we have any data after grouping
        if len(grouped_data) == 0:
            st.warning("No industry data available with current filters.")
            return
        
        # Sort by average compensation and get top 10
        grouped_data = grouped_data.sort_values('avg_compensation', ascending=False).head(10)
        
        # Create and display a simple bar chart
        fig = px.bar(
            grouped_data,
            x='industry_name',
            y='avg_compensation',
            labels={
                'industry_name': 'Industry',
                'avg_compensation': 'Average CEO Compensation'
            },
            title='Average CEO Compensation by Industry',
            color='avg_compensation',
            color_continuous_scale=color_scale,
            template=template
        )
        
        # Format y-axis as currency
        fig.update_yaxes(tickprefix='$', tickformat=',')
        # rotate y-axis labels
        fig.update_layout(xaxis_tickangle=-45)
        
        # Improve text visibility
        fig.update_layout(
            title_font=dict(size=20, color="white"),
            legend_title_font=dict(size=14, color="white"),
            legend_font=dict(size=12, color="white"),
            xaxis_title_font=dict(size=14, color="white"),
            yaxis_title_font=dict(size=14, color="white"),
            coloraxis_showscale=False,
            margin=dict(t=50, b=50, l=50, r=50),
            height=500
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
    
    # Simple box plot directly in the component
    #with tab2:
    #    pass
        # st.subheader("CEO Compensation Distribution by Firm Size")
        
        # # Check if firm_size column exists and has data
        # if 'firm_size' not in filtered_data.columns or filtered_data['firm_size'].nunique() < 1:
        #     st.warning("Firm size data not available with current filters.")
        # else:
        #     # Create box plot
        #     fig = px.box(
        #         filtered_data,
        #         x='firm_size',
        #         y='tdc1',
        #         labels={
        #             'firm_size': 'Firm Size',
        #             'tdc1': 'CEO Compensation'
        #         },
        #         title='CEO Compensation Distribution by Firm Size',
        #         color='firm_size',
        #         color_discrete_sequence=px.colors.qualitative.Vivid,
        #         template=template
        #     )
            
        #     # Format y-axis as currency
        #     fig.update_yaxes(tickprefix='$', tickformat=',')
        #     # sort x-axis by firm_size
        #     fig.update_xaxes(categoryorder='total ascending')
            
        #     # Improve text visibility
        #     fig.update_layout(
        #         title_font=dict(size=20, color="white"),
        #         legend_title_font=dict(size=14, color="white"),
        #         legend_font=dict(size=12, color="white"),
        #         xaxis_title_font=dict(size=14, color="white"),
        #         yaxis_title_font=dict(size=14, color="white"),
        #         margin=dict(t=50, b=50, l=50, r=50),
        #         height=500
        #     )
            
        #     # Display the chart
        #     st.plotly_chart(fig, use_container_width=True)
    
    # Simple trend chart
    with tab3:
        st.subheader("CEO Compensation Trends Over Time")
        
        # Check if we have enough years for a trend
        if filtered_data['year'].nunique() < 2:
            st.warning("Insufficient data for trend analysis. Trend analysis requires data from multiple years.")
        else:
            # Group by year and calculate average compensation
            trend_data = filtered_data.groupby('year')['tdc1'].mean().reset_index()
            
            # Create line chart
            fig = px.line(
                trend_data,
                x='year',
                y='tdc1',
                labels={
                    'year': 'Year',
                    'tdc1': 'Average Annual CEO Compensation'
                },
                title='Average CEO Compensation Over Time',
                template=template
            )
            
            # Format y-axis as currency
            fig.update_yaxes(tickprefix='$', tickformat=',')
            
            # Improve line visibility
            fig.update_traces(
                line=dict(color=line_color, width=3),
                marker=dict(size=10, color=line_color)
            )
            
            # Improve text visibility
            fig.update_layout(
                title_font=dict(size=20, color="white"),
                legend_title_font=dict(size=14, color="white"),
                legend_font=dict(size=12, color="white"),
                xaxis_title_font=dict(size=14, color="white"),
                yaxis_title_font=dict(size=14, color="white"),
                margin=dict(t=50, b=50, l=50, r=50),
                height=500
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
    
    # Simple performance comparison
    # with tab4:
        st.subheader("CEO Compensation vs ROA")
        
        # Check if we have ROA data
        if 'roa' not in filtered_data.columns or filtered_data['roa'].isna().all():
            st.warning("ROA data not available with current filters.")
        else:
            try:
                # Create performance quartiles
                filtered_data['roa_quartile'] = pd.qcut(
                    filtered_data['roa'], 
                    4, 
                    labels=['Bottom 25%', 'Lower Middle', 'Upper Middle', 'Top 25%']
                )
                
                # Group by quartile and calculate average compensation
                performance_data = filtered_data.groupby('roa_quartile')['tdc1'].mean().reset_index()
                
                # Create bar chart
                fig = px.bar(
                    performance_data,
                    x='roa_quartile',
                    y='tdc1',
                    labels={
                        'roa_quartile': 'ROA Performance Quartile',
                        'tdc1': 'Average Annual CEO Compensation'
                    },
                    title='Average CEO Compensation by ROA Performance',
                    category_orders={
                        'roa_quartile': ['Bottom 25%', 'Lower Middle', 'Upper Middle', 'Top 25%']
                    },
                    color='tdc1',
                    color_continuous_scale=color_scale,
                    template=template
                )
                
                # Format y-axis as currency
                fig.update_yaxes(tickprefix='$', tickformat=',')
                
                # Improve text visibility
                fig.update_layout(
                    title_font=dict(size=20, color="white"),
                    legend_title_font=dict(size=14, color="white"),
                    legend_font=dict(size=12, color="white"),
                    xaxis_title_font=dict(size=14, color="white"),
                    yaxis_title_font=dict(size=14, color="white"),
                    coloraxis_showscale=False,
                    margin=dict(t=50, b=50, l=50, r=50),
                    height=500
                )
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
            except ValueError as e:
                # Handle case where there are not enough distinct values for quartiles
                st.warning("Insufficient unique ROA values to create performance quartiles. Try adjusting filters to include more diverse data.")
                
                # Alternative: Show simple scatter plot instead
                st.subheader("CEO Compensation vs ROA (Scatter Plot)")
                
                # Create scatter plot
                fig = px.scatter(
                    filtered_data,
                    x='roa',
                    y='tdc1',
                    labels={
                        'roa': 'Return on Assets (ROA) %',
                        'tdc1': 'Annual CEO Compensation'
                    },
                    title='CEO Compensation vs ROA',
                    trendline="ols",
                    template=template,
                    color_discrete_sequence=[line_color]
                )
                
                # Format y-axis as currency
                fig.update_yaxes(tickprefix='$', tickformat=',')
                
                # Improve text visibility
                fig.update_layout(
                    title_font=dict(size=20, color="white"),
                    legend_title_font=dict(size=14, color="white"),
                    legend_font=dict(size=12, color="white"),
                    xaxis_title_font=dict(size=14, color="white"),
                    yaxis_title_font=dict(size=14, color="white"),
                    margin=dict(t=50, b=50, l=50, r=50),
                    height=500
                )
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
