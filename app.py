"""
Network Performance Anomaly Detection Dashboard
Streamlit Application for Visualizing Anomaly Detection Results

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Network Anomaly Detection Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .anomaly-high {
        background-color: #ff4b4b;
        color: white;
    }
    .anomaly-medium {
        background-color: #ffa500;
        color: white;
    }
    .anomaly-low {
        background-color: #00ff00;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📡 Network Performance Anomaly Detection Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Daily Signal Analysis - Egypt Regions")

# Sidebar
st.sidebar.header("Dashboard Settings")
st.sidebar.markdown("---")

# Load data function
@st.cache_data
def load_data():
    """Load the processed anomaly data."""
    # Look for processed data in same directory
    data_path = os.path.join(os.path.dirname(__file__), 'processed_data_with_anomalies.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['aggregate_date'] = pd.to_datetime(df['aggregate_date'])
        return df
    else:
        # Load raw data and process
        raw_path = os.path.join(os.path.dirname(__file__), 'Performance.csv')
        df = pd.read_csv(raw_path)
        df['aggregate_date'] = pd.to_datetime(df['aggregate_date'])
        # Filter
        df = df[df['sample_count'] >= 5]
        df = df[df['place_type'] == 'locality']
        df = df[df['aggregation_period'] == 'Day']
        return df

# Load anomaly log
@st.cache_data
def load_anomaly_log():
    """Load the anomaly log."""
    log_path = os.path.join(os.path.dirname(__file__), 'anomaly_log.csv')
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df['aggregate_date'] = pd.to_datetime(df['aggregate_date'])
        return df
    return None

# Load data
try:
    df = load_data()
    anomaly_log = load_anomaly_log()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False
    df = None
    anomaly_log = None

if data_loaded and df is not None:
    # Sidebar filters
    st.sidebar.subheader("Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(df['aggregate_date'].min().date(), df['aggregate_date'].max().date()),
        min_value=df['aggregate_date'].min().date(),
        max_value=df['aggregate_date'].max().date()
    )
    
    # Region filter
    regions = ['All'] + list(df['region'].unique())
    selected_region = st.sidebar.selectbox("Select Region", regions)
    
    # Carrier filter
    carriers = ['All'] + list(df['carrier_name'].unique())
    selected_carrier = st.sidebar.selectbox("Select Carrier", carriers)
    
    # Technology filter
    technologies = ['All'] + list(df['technology_type'].unique())
    selected_tech = st.sidebar.selectbox("Select Technology", technologies)
    
    # Apply filters
    df_filtered = df.copy()
    if len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered['aggregate_date'].dt.date >= date_range[0]) &
            (df_filtered['aggregate_date'].dt.date <= date_range[1])
        ]
    if selected_region != 'All':
        df_filtered = df_filtered[df_filtered['region'] == selected_region]
    if selected_carrier != 'All':
        df_filtered = df_filtered[df_filtered['carrier_name'] == selected_carrier]
    if selected_tech != 'All':
        df_filtered = df_filtered[df_filtered['technology_type'] == selected_tech]
    
    # Key Metrics Row
    st.markdown("## 📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Records",
            value=f"{len(df_filtered):,}",
            delta=f"Filtered from {len(df):,}"
        )
    
    with col2:
        if 'iso_anomaly' in df_filtered.columns:
            anomaly_count = df_filtered['iso_anomaly'].sum()
            anomaly_rate = df_filtered['iso_anomaly'].mean() * 100
            st.metric(
                label="Anomalies Detected",
                value=f"{anomaly_count:,}",
                delta=f"{anomaly_rate:.1f}% rate"
            )
        else:
            st.metric(label="Anomalies Detected", value="N/A")
    
    with col3:
        avg_download = df_filtered['mean_download_kbps'].mean() / 1000
        st.metric(
            label="Avg Download Speed",
            value=f"{avg_download:.1f} Mbps"
        )
    
    with col4:
        avg_latency = df_filtered['mean_latency_ms'].mean()
        st.metric(
            label="Avg Latency",
            value=f"{avg_latency:.1f} ms"
        )
    
    st.markdown("---")
    
    # Tab navigation
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview", 
        "🔍 Anomaly Detection", 
        "📊 Q&A Results",
        "🏆 Top 5 Worst",
        "📋 Anomaly Log",
        "ℹ️ Method Comparison"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.markdown("### Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Performance Distribution")
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Download speed distribution
            axes[0].hist(df_filtered['mean_download_kbps']/1000, bins=30, edgecolor='black', alpha=0.7, color='blue')
            axes[0].set_xlabel('Download Speed (Mbps)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Download Speed Distribution')
            
            # Latency distribution
            axes[1].hist(df_filtered['mean_latency_ms'], bins=30, edgecolor='black', alpha=0.7, color='red')
            axes[1].set_xlabel('Latency (ms)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Latency Distribution')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### Records by Region")
            region_counts = df_filtered['region'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(region_counts.index, region_counts.values, color='steelblue', alpha=0.7)
            ax.set_xlabel('Number of Records')
            ax.set_title('Records by Region')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Daily trends
        st.markdown("#### Daily Performance Trends")
        daily_avg = df_filtered.groupby('aggregate_date').agg({
            'mean_download_kbps': 'mean',
            'mean_latency_ms': 'mean'
        }).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(12, 5))
        
        ax1.plot(daily_avg['aggregate_date'], daily_avg['mean_download_kbps']/1000, 'b-', linewidth=2, marker='o', label='Download Speed')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Download Speed (Mbps)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2 = ax1.twinx()
        ax2.plot(daily_avg['aggregate_date'], daily_avg['mean_latency_ms'], 'r-', linewidth=2, marker='s', label='Latency')
        ax2.set_ylabel('Latency (ms)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title('Daily Average Performance')
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))
        plt.tight_layout()
        st.pyplot(fig)
    
    # Tab 2: Anomaly Detection
    with tab2:
        st.markdown("### Anomaly Detection Results")
        
        if 'iso_anomaly' in df_filtered.columns:
            # Method comparison
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Isolation Forest")
                iso_count = df_filtered['iso_anomaly'].sum()
                iso_rate = df_filtered['iso_anomaly'].mean() * 100
                st.metric("Anomalies", f"{iso_count}", f"{iso_rate:.1f}%")
                
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(df_filtered['iso_score'], bins=30, edgecolor='black', alpha=0.7)
                ax.axvline(x=0, color='red', linestyle='--', label='Threshold')
                ax.set_title('IF Score Distribution')
                ax.legend()
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### One-Class SVM")
                ocsvm_count = df_filtered['ocsvm_anomaly'].sum()
                ocsvm_rate = df_filtered['ocsvm_anomaly'].mean() * 100
                st.metric("Anomalies", f"{ocsvm_count}", f"{ocsvm_rate:.1f}%")
                
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(df_filtered['ocsvm_score'], bins=30, edgecolor='black', alpha=0.7, color='orange')
                ax.axvline(x=0, color='red', linestyle='--', label='Threshold')
                ax.set_title('OC-SVM Score Distribution')
                ax.legend()
                st.pyplot(fig)
            
            with col3:
                st.markdown("#### Autoencoder")
                ae_count = df_filtered['ae_anomaly'].sum()
                ae_rate = df_filtered['ae_anomaly'].mean() * 100
                st.metric("Anomalies", f"{ae_count}", f"{ae_rate:.1f}%")
                
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(df_filtered['ae_mse'], bins=30, edgecolor='black', alpha=0.7, color='green')
                ax.axvline(x=df_filtered['ae_threshold'].iloc[0], color='red', linestyle='--', label='Threshold')
                ax.set_title('AE MSE Distribution')
                ax.legend()
                st.pyplot(fig)
            
            # Anomaly scatter plot
            st.markdown("#### Anomalies in Feature Space")
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            scatter1 = axes[0].scatter(df_filtered['mean_download_kbps']/1000, 
                                       df_filtered['mean_latency_ms'],
                                       c=df_filtered['iso_anomaly'], cmap='coolwarm', alpha=0.6)
            axes[0].set_xlabel('Download Speed (Mbps)')
            axes[0].set_ylabel('Latency (ms)')
            axes[0].set_title('Isolation Forest')
            plt.colorbar(scatter1, ax=axes[0], label='Anomaly')
            
            scatter2 = axes[1].scatter(df_filtered['mean_download_kbps']/1000,
                                       df_filtered['mean_latency_ms'],
                                       c=df_filtered['ocsvm_anomaly'], cmap='coolwarm', alpha=0.6)
            axes[1].set_xlabel('Download Speed (Mbps)')
            axes[1].set_ylabel('Latency (ms)')
            axes[1].set_title('One-Class SVM')
            plt.colorbar(scatter2, ax=axes[1], label='Anomaly')
            
            scatter3 = axes[2].scatter(df_filtered['mean_download_kbps']/1000,
                                       df_filtered['mean_latency_ms'],
                                       c=df_filtered['ae_anomaly'], cmap='coolwarm', alpha=0.6)
            axes[2].set_xlabel('Download Speed (Mbps)')
            axes[2].set_ylabel('Latency (ms)')
            axes[2].set_title('Autoencoder')
            plt.colorbar(scatter3, ax=axes[2], label='Anomaly')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Anomaly detection results not available. Please run the Jupyter notebook first.")
    
    # Tab 3: Q&A Results
    with tab3:
        st.markdown("### Project Questions Answered")
        
        # Q1
        st.markdown("#### Q1: Which regions show the most variable download speed?")
        regional_stats = df_filtered.groupby('region').agg({
            'mean_download_kbps': ['mean', 'std']
        })
        regional_stats.columns = ['Mean Download', 'Std Download']
        regional_stats['CV (%)'] = regional_stats['Std Download'] / regional_stats['Mean Download'] * 100
        regional_stats = regional_stats.sort_values('CV (%)', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(regional_stats.index, regional_stats['CV (%)'], color='steelblue', alpha=0.7)
        ax.set_xlabel('Coefficient of Variation (%)')
        ax.set_title('Download Speed Variability by Region')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("**Answer:** Regions with highest CV show the most variable performance.")
        
        st.markdown("---")
        
        # Q2
        st.markdown("#### Q2: Days with abnormal performance across multiple regions?")
        if 'iso_anomaly' in df_filtered.columns:
            daily_anomalies = df_filtered.groupby('aggregate_date').agg({
                'iso_anomaly': 'sum',
                'mean_download_kbps': 'mean',
                'mean_latency_ms': 'mean'
            }).reset_index()
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(daily_anomalies['aggregate_date'], daily_anomalies['iso_anomaly'], color='purple', alpha=0.7)
            ax.set_xlabel('Date')
            ax.set_ylabel('Anomaly Count')
            ax.set_title('Daily Anomaly Count Across All Regions')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Q3
        st.markdown("#### Q3: Carrier performance consistency across regions?")
        carrier_region = df_filtered.groupby(['carrier_name', 'region'])['mean_download_kbps'].mean().unstack()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(carrier_region/1000, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Download (Mbps)'})
        ax.set_title('Carrier Performance by Region (Download Speed)')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Q4
        st.markdown("#### Q4: LTE vs 5G anomaly patterns?")
        if 'iso_anomaly' in df_filtered.columns:
            tech_comparison = df_filtered.groupby('technology_type').agg({
                'iso_anomaly': ['sum', 'mean'],
                'mean_download_kbps': 'mean',
                'mean_latency_ms': 'mean'
            })
            tech_comparison.columns = ['Anomaly Count', 'Anomaly Rate', 'Avg Download', 'Avg Latency']
            tech_comparison['Anomaly Rate'] = tech_comparison['Anomaly Rate'] * 100
            
            st.dataframe(tech_comparison.round(2), use_container_width=True)
        
        st.markdown("---")
        
        # Q5
        st.markdown("#### Q5: Top 5 Worst Performers")
        if 'worst_score' in df_filtered.columns:
            top5 = df_filtered.nlargest(5, 'worst_score')[
                ['aggregate_date', 'region', 'carrier_name', 'technology_type',
                 'mean_download_kbps', 'mean_latency_ms', 'worst_score']
            ]
            st.dataframe(top5, use_container_width=True)
    
    # Tab 4: Top 5 Worst
    with tab4:
        st.markdown("### 🏆 Top 5 Worst Performing Combinations")
        
        if 'worst_score' in df_filtered.columns:
            # Define criteria explanation
            st.markdown("""
            **Worst Score Definition:**
            - Download Speed: 40% weight (lower speed = higher score)
            - Upload Speed: 20% weight (lower speed = higher score)
            - Latency: 30% weight (higher latency = higher score)
            - Anomaly Flags: 10% weight (more flags = higher score)
            """)
            
            top5 = df_filtered.nlargest(5, 'worst_score')
            
            for i, (_, row) in enumerate(top5.iterrows(), 1):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    **#{i} - {row['region']} - {row['carrier_name']}**
                    - Date: {row['aggregate_date'].strftime('%Y-%m-%d')}
                    - Technology: {row['technology_type']}
                    - Download: {row['mean_download_kbps']/1000:.2f} Mbps
                    - Upload: {row['mean_upload_kbps']/1000:.2f} Mbps
                    - Latency: {row['mean_latency_ms']:.1f} ms
                    """)
                
                with col2:
                    st.metric("Worst Score", f"{row['worst_score']:.4f}")
                
                st.markdown("---")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 5))
            labels = [f"{row['region']}\n{row['carrier_name']}\n{row['aggregate_date'].strftime('%m/%d')}" for _, row in top5.iterrows()]
            ax.barh(range(5), top5['worst_score'].values, color='red', alpha=0.7)
            ax.set_yticks(range(5))
            ax.set_yticklabels(labels)
            ax.set_xlabel('Worst Score')
            ax.set_title('Top 5 Worst Performers')
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Worst score calculation not available. Please run the Jupyter notebook first.")
    
    # Tab 5: Anomaly Log
    with tab5:
        st.markdown("### 📋 Anomaly Log")
        
        if anomaly_log is not None and len(anomaly_log) > 0:
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                method_filter = st.multiselect(
                    "Filter by Detection Method",
                    ['IsolationForest', 'OC-SVM', 'Autoencoder'],
                    default=['IsolationForest', 'OC-SVM', 'Autoencoder']
                )
            
            with col2:
                min_methods = st.slider("Minimum methods agreement", 1, 3, 1)
            
            with col3:
                show_count = st.number_input("Show top N records", 10, 1000, 50)
            
            # Apply filters
            filtered_log = anomaly_log[
                (anomaly_log['methods_count'] >= min_methods) &
                (anomaly_log['detection_methods'].apply(lambda x: any(m in x for m in method_filter) if method_filter else True))
            ].head(show_count)
            
            st.markdown(f"**Showing {len(filtered_log)} of {len(anomaly_log)} anomalies**")
            
            # Display dataframe
            display_cols = ['aggregate_date', 'region', 'carrier_name', 'technology_type',
                           'mean_download_kbps', 'mean_upload_kbps', 'mean_latency_ms',
                           'detection_methods', 'methods_count']
            
            st.dataframe(
                filtered_log[display_cols].style.background_gradient(subset=['methods_count'], cmap='Reds'),
                use_container_width=True
            )
            
            # Download button
            csv = filtered_log.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Filtered Log",
                csv,
                "anomaly_log_filtered.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.warning("Anomaly log not available. Please run the Jupyter notebook first.")
    
    # Tab 6: Method Comparison
    with tab6:
        st.markdown("### ℹ️ Anomaly Detection Methods Comparison")
        
        if 'iso_anomaly' in df_filtered.columns:
            # Summary table
            comparison_data = {
                'Method': ['Isolation Forest', 'One-Class SVM', 'Autoencoder'],
                'Anomalies Detected': [
                    df_filtered['iso_anomaly'].sum(),
                    df_filtered['ocsvm_anomaly'].sum(),
                    df_filtered['ae_anomaly'].sum()
                ],
                'Anomaly Rate (%)': [
                    f"{df_filtered['iso_anomaly'].mean()*100:.2f}",
                    f"{df_filtered['ocsvm_anomaly'].mean()*100:.2f}",
                    f"{df_filtered['ae_anomaly'].mean()*100:.2f}"
                ],
                'Processing Speed': ['Fast', 'Slow', 'Moderate'],
                'Best For': [
                    'High-dimensional data',
                    'Complex boundaries',
                    'Non-linear patterns'
                ]
            }
            
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            
            st.markdown("#### Agreement Analysis")
            
            if 'all_methods' in df_filtered.columns:
                agreement = {
                    'All 3 Methods Agree': (df_filtered['all_methods'] == 3).sum(),
                    'Exactly 2 Methods': (df_filtered['all_methods'] == 2).sum(),
                    'Exactly 1 Method': (df_filtered['all_methods'] == 1).sum(),
                    'No Method': (df_filtered['all_methods'] == 0).sum()
                }
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(agreement.keys(), agreement.values(), color=['red', 'orange', 'gray', 'green'], alpha=0.7)
                ax.set_ylabel('Record Count')
                ax.set_title('Method Agreement Analysis')
                plt.xticks(rotation=15)
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown("""
            #### Method Characteristics
            
            **1. Isolation Forest**
            - Fast training and prediction
            - Works well with high-dimensional data
            - Does not require scaling
            - Flags isolated points regardless of distribution
            
            **2. One-Class SVM**
            - Learns a decision boundary around normal data
            - Sensitive to kernel choice
            - Computationally expensive for large datasets
            - Good for capturing complex boundaries
            
            **3. Autoencoder**
            - Learns to reconstruct normal patterns
            - Captures non-linear relationships
            - Requires more training time
            - Good for detecting subtle deviations
            """)
        else:
            st.warning("Method comparison not available. Please run the Jupyter notebook first.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>📡 Network Performance Anomaly Detection Dashboard | Project 05 - Daily Signal</p>
        <p>Methods: Isolation Forest | One-Class SVM | Autoencoder</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("No data available. Please ensure the data files are in the correct location.")
    st.markdown("""
    **Expected file locations:**
    - `data/Performance.csv` - Raw data
    - `outputs/processed_data_with_anomalies.csv` - Processed data with anomaly flags
    - `outputs/anomaly_log.csv` - Anomaly log file
    """)
