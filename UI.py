import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr

# Instructions for running
# Install required libraries before running:
# pip install streamlit pandas numpy matplotlib seaborn folium streamlit-folium scikit-learn

class GISDataAnalyzer:
    def __init__(self, df):
        """Initialize with a DataFrame"""
        self.df = df
        
        # Analysis methods
        self.analysis_methods = [
            "Linear Regression",
            "Pearson Correlation",
            "Feature Importance"
        ]
    
    def create_interactive_map(self, lat_col, lon_col, value_col):
        """Create an interactive Folium map"""
        try:
            # Validate data
            if self.df[lat_col].isnull().any() or self.df[lon_col].isnull().any():
                raise ValueError("Latitude or Longitude columns contain missing values.")
            
            # Calculate center of the map
            center_lat = self.df[lat_col].mean()
            center_lon = self.df[lon_col].mean()
            
            # Create map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
            
            # Color normalization
            min_val = self.df[value_col].min()
            max_val = self.df[value_col].max()
            if min_val == max_val:
                min_val -= 1  # Avoid zero range
            
            # Add marker cluster
            marker_cluster = MarkerCluster().add_to(m)
            
            # Add markers
            for idx, row in self.df.iterrows():
                # Normalize color
                normalized_value = (row[value_col] - min_val) / (max_val - min_val)
                color = self._get_color_gradient(normalized_value)
                
                # Create popup text
                popup_text = f"{value_col}: {row[value_col]:.2f}<br>"
                popup_text += "<br>".join([f"{col}: {row[col]}" for col in [lat_col, lon_col]])
                
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=5,
                    popup=popup_text,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(marker_cluster)
            
            return m
        except Exception as e:
            raise ValueError(f"Error creating map: {e}")
    
    def _get_color_gradient(self, value):
        """Generate color gradient from blue to red"""
        r = int(255 * value)
        b = int(255 * (1 - value))
        return f'rgb({r},0,{b})'
    
    def perform_analysis(self, primary_var, secondary_vars, method):
        """Perform statistical analysis"""
        try:
            if not secondary_vars:
                raise ValueError("No secondary variables selected for analysis.")
            
            # Filter numeric secondary variables
            numeric_vars = [var for var in secondary_vars if pd.api.types.is_numeric_dtype(self.df[var])]
            if not numeric_vars:
                raise ValueError("None of the selected secondary variables are numeric.")
            
            X = self.df[numeric_vars]
            y = self.df[primary_var]
            
            results = {}
            
            if method == "Linear Regression":
                model = LinearRegression()
                model.fit(X, y)
                results = {
                    'Coefficients': dict(zip(numeric_vars, model.coef_)),
                    'Intercept': model.intercept_,
                    'R-squared': model.score(X, y)
                }
            
            elif method == "Pearson Correlation":
                correlations = {}
                for var in numeric_vars:
                    corr, p_value = pearsonr(self.df[primary_var], self.df[var])
                    correlations[var] = {
                        'Correlation': corr, 
                        'P-value': p_value
                    }
                results = correlations
            
            elif method == "Feature Importance":
                model = LinearRegression()
                model.fit(X, y)
                
                # Use permutation importance
                perm_importance = permutation_importance(model, X, y, n_repeats=10)
                results = {
                    var: imp for var, imp in zip(numeric_vars, perm_importance.importances_mean)
                }
            
            return results
        except Exception as e:
            raise ValueError(f"Error performing analysis: {e}")
    
    def create_correlation_heatmap(self, columns):
        """Create correlation heatmap"""
        try:
            numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(self.df[col])]
            if len(numeric_cols) < 2:
                raise ValueError("Select at least two numeric columns for the heatmap.")
            
            plt.figure(figsize=(10, 8))
            correlation_matrix = self.df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            return plt
        except Exception as e:
            raise ValueError(f"Error creating heatmap: {e}")

def main():
    # Set page configuration
    st.set_page_config(page_title="GIS Data Analyzer", layout="wide")
    
    # Title
    st.title("🌍 GIS Data Analysis Tool")
    
    # Sidebar for file upload
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Initialize analyzer
            analyzer = GISDataAnalyzer(df)
            
            # Display data overview
            st.header("Data Overview")
            st.dataframe(df)
            
            # Sidebar for analysis configuration
            st.sidebar.header("Analysis Configuration")
            
            # Columns for selection
            col_names = df.columns.tolist()
            
            # Spatial columns selection
            st.sidebar.subheader("Spatial Columns")
            lat_col = st.sidebar.selectbox("Latitude Column", col_names)
            lon_col = st.sidebar.selectbox("Longitude Column", col_names)
            value_col = st.sidebar.selectbox("Value Column for Mapping", col_names)
            
            # Analysis settings
            st.sidebar.subheader("Statistical Analysis")
            primary_var = st.sidebar.selectbox("Primary Variable", col_names)
            secondary_vars = st.sidebar.multiselect("Secondary Variables", col_names)
            analysis_method = st.sidebar.selectbox("Analysis Method", analyzer.analysis_methods)
            
            # Tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Interactive Map", "Statistical Analysis", "Correlation Heatmap"])
            
            with tab1:
                st.header("Interactive Geospatial Map")
                try:
                    map_obj = analyzer.create_interactive_map(lat_col, lon_col, value_col)
                    folium_static(map_obj)
                except Exception as e:
                    st.error(e)
            
            with tab2:
                st.header("Statistical Analysis Results")
                if secondary_vars:
                    try:
                        results = analyzer.perform_analysis(primary_var, secondary_vars, analysis_method)
                        st.write(f"Analysis Method: {analysis_method}")
                        st.dataframe(pd.DataFrame.from_dict(results, orient='index'))
                    except Exception as e:
                        st.error(e)
                else:
                    st.warning("Please select secondary variables for analysis.")
            
            with tab3:
                st.header("Correlation Heatmap")
                try:
                    if len(secondary_vars) > 1:
                        heatmap = analyzer.create_correlation_heatmap(secondary_vars + [primary_var])
                        st.pyplot(heatmap)
                    else:
                        st.warning("Select multiple variables to generate a correlation heatmap.")
                except Exception as e:
                    st.error(e)
        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
