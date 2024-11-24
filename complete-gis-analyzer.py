import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

class GISDataAnalyzer:
    def __init__(self, df):
        """Initialize with a DataFrame"""
        self.df = df
        
        # Expanded analysis methods
        self.analysis_methods = [
            "Linear Regression",
            "Pearson Correlation",
            "Lasso Regression",
            "Elastic Net Regression",
            "Random Forest",
            "Neural Network",
            "Gradient Boosting Machine"
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
    
    def _prepare_data(self, X, y):
        """Prepare data for modeling by scaling features"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y
    
    def perform_analysis(self, primary_var, secondary_vars, method):
        """Perform statistical analysis with expanded methods"""
        try:
            if not secondary_vars:
                raise ValueError("No secondary variables selected for analysis.")
            
            # Filter numeric secondary variables
            numeric_vars = [var for var in secondary_vars if pd.api.types.is_numeric_dtype(self.df[var])]
            if not numeric_vars:
                raise ValueError("None of the selected secondary variables are numeric.")
            
            X = self.df[numeric_vars]
            y = self.df[primary_var]
            
            # Prepare data
            X_scaled, y = self._prepare_data(X, y)
            
            # Split data for training and testing
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            results = {}
            
            if method == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                results = {
                    'Coefficients': dict(zip(numeric_vars, model.coef_)),
                    'Intercept': model.intercept_,
                    'R-squared (train)': model.score(X_train, y_train),
                    'R-squared (test)': model.score(X_test, y_test)
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
            
            elif method == "Lasso Regression":
                model = Lasso(alpha=0.1, random_state=42)
                model.fit(X_train, y_train)
                results = {
                    'Coefficients': dict(zip(numeric_vars, model.coef_)),
                    'Intercept': model.intercept_,
                    'R-squared (train)': model.score(X_train, y_train),
                    'R-squared (test)': model.score(X_test, y_test)
                }
            
            elif method == "Elastic Net Regression":
                model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
                model.fit(X_train, y_train)
                results = {
                    'Coefficients': dict(zip(numeric_vars, model.coef_)),
                    'Intercept': model.intercept_,
                    'R-squared (train)': model.score(X_train, y_train),
                    'R-squared (test)': model.score(X_test, y_test)
                }
            
            elif method == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                results = {
                    'Feature Importance': dict(zip(numeric_vars, model.feature_importances_)),
                    'R-squared (train)': model.score(X_train, y_train),
                    'R-squared (test)': model.score(X_test, y_test)
                }
            
            elif method == "Neural Network":
                model = MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    max_iter=1000,
                    random_state=42
                )
                model.fit(X_train, y_train)
                results = {
                    'R-squared (train)': model.score(X_train, y_train),
                    'R-squared (test)': model.score(X_test, y_test),
                    'Number of iterations': model.n_iter_,
                    'Loss': model.loss_
                }
            
            elif method == "Gradient Boosting Machine":
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    random_state=42
                )
                model.fit(X_train, y_train)
                results = {
                    'Feature Importance': dict(zip(numeric_vars, model.feature_importances_)),
                    'R-squared (train)': model.score(X_train, y_train),
                    'R-squared (test)': model.score(X_test, y_test)
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
                        st.write("Results:")
                        
                        # Format results based on method
                        if isinstance(results, dict):
                            for key, value in results.items():
                                if isinstance(value, dict):
                                    st.write(f"\n{key}:")
                                    for subkey, subvalue in value.items():
                                        st.write(f"- {subkey}: {subvalue:.4f}")
                                else:
                                    st.write(f"- {key}: {value:.4f}")
                        else:
                            st.write(results)
                            
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
