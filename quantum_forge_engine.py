import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import os
import json
import tempfile
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import time
import logging
from pathlib import Path
import io
import base64
from datetime import datetime
import uuid
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Import DataSynthesizer with new names
from DataSynthesizer.DataDescriber import DataDescriber as MatrixDescriber
from DataSynthesizer.DataGenerator import DataGenerator as MatrixGenerator
from DataSynthesizer.ModelInspector import ModelInspector as PatternAnalyzer
from DataSynthesizer.lib.utils import read_json_file as parse_json_config

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Quantum Forge Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create temp directory for storing files
TEMP_DIR = Path(tempfile.mkdtemp())

# Define constants
APP_VERSION = "3.0.0"
DEFAULT_EPSILON = 1.0
DEFAULT_NUM_TUPLES = 5000
DEFAULT_THRESHOLD = 20
DEFAULT_DEGREE = 2
CACHE_TTL = 3600  # Cache time-to-live in seconds

# Initialize session state if not already done
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'cache' not in st.session_state:
    st.session_state.cache = {}
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Upload"
if 'error_log' not in st.session_state:
    st.session_state.error_log = []

# Utility classes for better code organization

class Cache:
    """Simple caching mechanism to improve performance."""
    
    @staticmethod
    def get(key: str) -> Any:
        """Get a value from the cache."""
        if key in st.session_state.cache:
            value, timestamp = st.session_state.cache[key]
            if time.time() - timestamp < CACHE_TTL:
                return value
            else:
                # Expired
                del st.session_state.cache[key]
        return None
    
    @staticmethod
    def set(key: str, value: Any) -> None:
        """Set a value in the cache."""
        st.session_state.cache[key] = (value, time.time())
    
    @staticmethod
    def clear() -> None:
        """Clear the cache."""
        st.session_state.cache = {}


class DataProcessor:
    """Utility class for data preprocessing and quality assessment."""
    
    @staticmethod
    def detect_column_types(df: pd.DataFrame, threshold: int = 20) -> Dict[str, bool]:
        """Automatically detect column types based on unique value count."""
        column_types = {}
        for col in df.columns:
            unique_count = df[col].nunique()
            is_categorical = unique_count < threshold
            column_types[col] = is_categorical
        return column_types
    
    @staticmethod
    def detect_candidate_keys(df: pd.DataFrame) -> Dict[str, bool]:
        """Detect potential candidate keys based on uniqueness."""
        candidate_keys = {}
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            candidate_keys[col] = unique_ratio > 0.9  # If more than 90% values are unique
        return candidate_keys
    
    @staticmethod
    def get_data_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality metrics for a dataframe."""
        metrics = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "missing_percentage": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            "duplicate_rows": df.duplicated().sum(),
            "column_metrics": {}
        }
        
        for col in df.columns:
            col_metrics = {
                "type": str(df[col].dtype),
                "unique_values": df[col].nunique(),
                "missing_values": df[col].isnull().sum(),
                "missing_percentage": (df[col].isnull().sum() / len(df)) * 100
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_metrics.update({
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "std": df[col].std()
                })
            
            metrics["column_metrics"][col] = col_metrics
        
        return metrics
    
    @staticmethod
    def compare_datasets(original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Compare original and synthetic datasets and calculate similarity metrics."""
        comparison = {
            "column_comparisons": {},
            "overall_similarity": {}
        }
        
        # Compare common columns
        common_columns = set(original_df.columns) & set(synthetic_df.columns)
        
        # Overall metrics
        comparison["overall_similarity"]["column_count_match"] = len(original_df.columns) == len(synthetic_df.columns)
        comparison["overall_similarity"]["columns_match"] = set(original_df.columns) == set(synthetic_df.columns)
        
        # Calculate Jensen-Shannon divergence for distributions
        from scipy.spatial.distance import jensenshannon
        from scipy.stats import wasserstein_distance
        
        for col in common_columns:
            col_comparison = {}
            
            if pd.api.types.is_numeric_dtype(original_df[col]) and pd.api.types.is_numeric_dtype(synthetic_df[col]):
                # For numeric columns
                col_comparison["mean_difference"] = abs(original_df[col].mean() - synthetic_df[col].mean())
                col_comparison["std_difference"] = abs(original_df[col].std() - synthetic_df[col].std())
                col_comparison["min_difference"] = abs(original_df[col].min() - synthetic_df[col].min())
                col_comparison["max_difference"] = abs(original_df[col].max() - synthetic_df[col].max())
                
                # Normalize data for distribution comparison
                orig_data = original_df[col].dropna()
                synth_data = synthetic_df[col].dropna()
                
                if len(orig_data) > 0 and len(synth_data) > 0:
                    # Calculate Wasserstein distance (Earth Mover's Distance)
                    try:
                        col_comparison["wasserstein_distance"] = wasserstein_distance(
                            orig_data, synth_data
                        )
                    except Exception as e:
                        col_comparison["wasserstein_distance"] = None
                        logger.warning(f"Could not calculate Wasserstein distance for {col}: {str(e)}")
                
            else:
                # For categorical columns
                orig_counts = original_df[col].value_counts(normalize=True).sort_index()
                synth_counts = synthetic_df[col].value_counts(normalize=True).sort_index()
                
                # Get all unique values
                all_values = sorted(set(orig_counts.index) | set(synth_counts.index))
                
                # Create full distributions with zeros for missing values
                orig_dist = pd.Series([orig_counts.get(val, 0) for val in all_values], index=all_values)
                synth_dist = pd.Series([synth_counts.get(val, 0) for val in all_values], index=all_values)
                
                # Calculate Jensen-Shannon divergence
                try:
                    col_comparison["js_divergence"] = jensenshannon(orig_dist, synth_dist)
                except Exception as e:
                    col_comparison["js_divergence"] = None
                    logger.warning(f"Could not calculate JS divergence for {col}: {str(e)}")
                
                # Calculate value preservation
                common_values = set(original_df[col].unique()) & set(synthetic_df[col].unique())
                col_comparison["value_preservation"] = len(common_values) / len(set(original_df[col].unique()))
            
            comparison["column_comparisons"][col] = col_comparison
        
        return comparison


class Visualizer:
    """Utility class for creating visualizations."""
    
    @staticmethod
    def create_histogram_comparison(original_df: pd.DataFrame, synthetic_df: pd.DataFrame, column: str) -> go.Figure:
        """Create a histogram comparison between original and synthetic data."""
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Original Data", "Synthetic Data"])
        
        # Original data histogram
        fig.add_trace(
            go.Histogram(
                x=original_df[column],
                name="Original",
                marker_color='#1f77b4',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Synthetic data histogram
        fig.add_trace(
            go.Histogram(
                x=synthetic_df[column],
                name="Synthetic",
                marker_color='#ff7f0e',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=f"Distribution Comparison: {column}",
            height=400,
            showlegend=False,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def create_bar_comparison(original_df: pd.DataFrame, synthetic_df: pd.DataFrame, column: str) -> go.Figure:
        """Create a bar chart comparison for categorical columns."""
        # Get value counts
        orig_counts = original_df[column].value_counts().sort_index()
        synth_counts = synthetic_df[column].value_counts().sort_index()
        
        # Get all unique values
        all_values = sorted(set(orig_counts.index) | set(synth_counts.index))
        
        # Create full distributions with zeros for missing values
        orig_dist = pd.Series([orig_counts.get(val, 0) for val in all_values], index=all_values)
        synth_dist = pd.Series([synth_counts.get(val, 0) for val in all_values], index=all_values)
        
        # Create figure
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Original Data", "Synthetic Data"])
        
        # Original data bar chart
        fig.add_trace(
            go.Bar(
                x=orig_dist.index,
                y=orig_dist.values,
                name="Original",
                marker_color='#1f77b4'
            ),
            row=1, col=1
        )
        
        # Synthetic data bar chart
        fig.add_trace(
            go.Bar(
                x=synth_dist.index,
                y=synth_dist.values,
                name="Synthetic",
                marker_color='#ff7f0e'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=f"Category Distribution Comparison: {column}",
            height=400,
            showlegend=False,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame, title: str) -> go.Figure:
        """Create a correlation heatmap for numeric columns."""
        # Calculate correlation matrix
        corr_matrix = df.select_dtypes(include=['number']).corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmin=-1, zmax=1,
            colorbar=dict(title='Correlation')
        ))
        
        fig.update_layout(
            title_text=title,
            height=600,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def create_scatter_comparison(original_df: pd.DataFrame, synthetic_df: pd.DataFrame, 
                                 x_col: str, y_col: str) -> go.Figure:
        """Create a scatter plot comparison between original and synthetic data."""
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Original Data", "Synthetic Data"])
        
        # Original data scatter
        fig.add_trace(
            go.Scatter(
                x=original_df[x_col],
                y=original_df[y_col],
                mode='markers',
                name="Original",
                marker=dict(color='#1f77b4', opacity=0.7)
            ),
            row=1, col=1
        )
        
        # Synthetic data scatter
        fig.add_trace(
            go.Scatter(
                x=synthetic_df[x_col],
                y=synthetic_df[y_col],
                mode='markers',
                name="Synthetic",
                marker=dict(color='#ff7f0e', opacity=0.7)
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=f"Relationship Comparison: {x_col} vs {y_col}",
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def create_boxplot_comparison(original_df: pd.DataFrame, synthetic_df: pd.DataFrame, column: str) -> go.Figure:
        """Create a box plot comparison between original and synthetic data."""
        fig = go.Figure()
        
        # Original data box plot
        fig.add_trace(
            go.Box(
                y=original_df[column],
                name="Original",
                marker_color='#1f77b4',
                boxmean=True
            )
        )
        
        # Synthetic data box plot
        fig.add_trace(
            go.Box(
                y=synthetic_df[column],
                name="Synthetic",
                marker_color='#ff7f0e',
                boxmean=True
            )
        )
        
        fig.update_layout(
            title_text=f"Distribution Comparison: {column}",
            height=500,
            template="plotly_white"
        )
        
        return fig


class UIHelper:
    """Helper class for UI components and styling."""
    
    @staticmethod
    def apply_custom_css() -> None:
        """Apply custom CSS styling to the app."""
        st.markdown("""
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: #1E3A8A;
        }
        .stButton>button {
            background-color: #1E3A8A;
            color: white;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #2563EB;
            border-color: #2563EB;
        }
        .stProgress .st-bo {
            background-color: #1E3A8A;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #F3F4F6;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1E3A8A;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_header() -> None:
        """Display the application header with logo and version info."""
        col1, col2 = st.columns([1, 5])
        
        with col1:
            st.image("https://raw.githubusercontent.com/DataResponsibly/DataSynthesizer/master/figures/DataSynthesizer.png", width=80)
        
        with col2:
            st.title("Advanced Data Synthesizer")
            st.markdown(f"<p style='margin-top:-15px;color:#666;'>Version {APP_VERSION} | Privacy-Preserving Synthetic Data Generation</p>", unsafe_allow_html=True)
    
    @staticmethod
    def display_footer() -> None:
        """Display the application footer."""
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align:center;color:#666;font-size:0.8em;">
                <p>Powered by <a href="https://github.com/DataResponsibly/DataSynthesizer" target="_blank">DataSynthesizer</a> | 
                <a href="jasma.xyz" target="_blank">Jasma</a> | 
                <a href="git@github.com:jasmadata/jasma-data-synthesizer-p1-v1.git" target="_blank">GitHub Repository</a></p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    @staticmethod
    def display_info_card(title: str, content: str, icon: str = "ℹ️") -> None:
        """Display an information card with custom styling."""
        st.markdown(
            f"""
            <div style="padding:1rem;background-color:#F0F9FF;border-radius:0.5rem;border-left:4px solid #0EA5E9;">
                <div style="display:flex;align-items:center;">
                    <div style="font-size:1.5rem;margin-right:0.5rem;">{icon}</div>
                    <div>
                        <h4 style="margin:0;color:#0369A1;">{title}</h4>
                        <p style="margin:0.5rem 0 0 0;color:#334155;">{content}</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    @staticmethod
    def display_metric_card(title: str, value: Any, delta: Optional[float] = None, 
                           help_text: Optional[str] = None) -> None:
        """Display a metric card with optional delta indicator."""
        col = st.column_config.NumberColumn(
            title,
            help=help_text,
            format="%d" if isinstance(value, int) else "%.2f"
        )
        
        if delta is not None:
            st.metric(title, value, delta=delta, help=help_text)
        else:
            st.metric(title, value, help=help_text)
    
    @staticmethod
    def create_download_link(object_to_download, download_filename, button_text):
        """Generate a download link for any object."""
        if isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)
            b64 = base64.b64encode(object_to_download.encode()).decode()
            mime_type = "text/csv"
        elif isinstance(object_to_download, dict):
            object_to_download = json.dumps(object_to_download, indent=4)
            b64 = base64.b64encode(object_to_download.encode()).decode()
            mime_type = "application/json"
        else:
            b64 = base64.b64encode(object_to_download.encode()).decode()
            mime_type = "text/plain"
        
        button_uuid = str(uuid.uuid4()).replace('-', '')
        button_id = re.sub('\d+', '', button_uuid)
        
        custom_css = f"""
            <style>
                #{button_id} {{
                    background-color: #1E3A8A;
                    color: white;
                    padding: 0.5rem 1rem;
                    border-radius: 0.25rem;
                    border: none;
                    font-size: 0.875rem;
                    font-weight: 500;
                    cursor: pointer;
                    margin: 0.25rem 0;
                }}
                #{button_id}:hover {{
                    background-color: #2563EB;
                }}
            </style>
        """
        
        dl_link = (
            custom_css +
            f'<a download="{download_filename}" id="{button_id}" href="data:{mime_type};base64,{b64}">{button_text}</a><br>'
        )
        
        return dl_link

class QuantumForgeEngine:
    """
    A Streamlit application for generating synthetic data using advanced quantum-inspired algorithms.
    """
    
    def __init__(self):
        """Initialize the application state and configuration."""
        self.input_df = None
        self.synthetic_df = None
        self.attribute_description = None
        self.description_file = TEMP_DIR / "matrix_config.json"
        self.synthetic_data_file = TEMP_DIR / "forged_data.csv"
        self.input_data_file = None
        self.categorical_attributes = {}
        self.candidate_keys = {}
        
    def configure_control_panel(self) -> None:
        """Render the sidebar with configuration options."""
        st.sidebar.title("Configuration")
        
        # Synthesis mode selection
        self.mode = st.sidebar.radio(
            "Select Synthesis Mode",
            ["Independent Attribute Mode", "Correlated Attribute Mode"],
            help="Independent mode treats each column separately. Correlated mode preserves relationships between columns."
        )
        
        # Privacy parameters
        st.sidebar.subheader("Privacy Settings")
        self.epsilon = st.sidebar.slider(
            "Epsilon (Privacy Budget)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Higher values mean less privacy but more accuracy. Set to 0 to disable differential privacy."
        )
        
        # Data generation parameters
        st.sidebar.subheader("Data Generation")
        self.num_tuples = st.sidebar.number_input(
            "Number of Records to Generate",
            min_value=1,
            max_value=1000000,
            value=5000,
            step=1000,
            help="Number of synthetic data records to generate"
        )
        
        self.threshold_value = st.sidebar.slider(
            "Categorical Attribute Threshold",
            min_value=1,
            max_value=100,
            value=20,
            help="Attributes with fewer unique values than this threshold will be treated as categorical"
        )
        
        # Only show Bayesian network degree for correlated mode
        if "Correlated" in self.mode:
            self.degree_of_bayesian_network = st.sidebar.slider(
                "Max Degree of Bayesian Network",
                min_value=1,
                max_value=10,
                value=2,
                help="Maximum number of parents in the Bayesian network"
            )
        
        # About section
        st.sidebar.subheader("About")
        st.sidebar.info(
            """
            This application uses advanced quantum-inspired algorithms to generate synthetic data 
            that preserves the statistical properties of the original dataset 
            while protecting privacy.
            
            [GitHub Repository](git@github.com:jasmadata/jasma-data-synthesizer-p1-v1.git)
            """
        )
    
    def ingest_source_data(self) -> None:
        """Handle file upload and initial data processing."""
        st.title("Quantum Forge Engine")
        
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            help="Upload your dataset in CSV format"
        )
        
        if uploaded_file is not None:
            try:
                # Load and display the data
                self.input_df = pd.read_csv(uploaded_file)
                
                # Save the uploaded file to a temporary location
                self.input_data_file = TEMP_DIR / uploaded_file.name
                self.input_df.to_csv(self.input_data_file, index=False)
                
                st.success(f"Successfully loaded dataset with {self.input_df.shape[0]} rows and {self.input_df.shape[1]} columns")
                
                # Display data preview
                with st.expander("Preview Input Data", expanded=True):
                    st.dataframe(self.input_df.head(10))
                    
                    # Display data statistics
                    st.subheader("Data Statistics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Numeric Columns Summary")
                        st.dataframe(self.input_df.describe())
                    with col2:
                        st.write("Missing Values")
                        missing_data = pd.DataFrame({
                            'Column': self.input_df.columns,
                            'Missing Values': self.input_df.isnull().sum().values,
                            'Percentage': round(self.input_df.isnull().sum().values / len(self.input_df) * 100, 2)
                        })
                        st.dataframe(missing_data)
                
                # Configure column types
                self.define_attribute_schema()
                
                return True
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                logger.error(f"Error loading data: {str(e)}", exc_info=True)
                return False
        return False
    
    def define_attribute_schema(self) -> None:
        """Allow users to configure column types and candidate keys."""
        st.subheader("Define Attribute Schema")
        
        with st.expander("Attribute Configuration", expanded=True):
            st.info("Select categorical attributes and unique identifiers")
            
            # Automatically detect categorical columns based on threshold
            detected_categorical = {}
            for col in self.input_df.columns:
                unique_count = self.input_df[col].nunique()
                is_categorical = unique_count < self.threshold_value
                detected_categorical[col] = is_categorical
            
            # Allow user to override
            st.write("Categorical Attributes")
            cols_per_row = 3
            col_groups = [self.input_df.columns[i:i+cols_per_row] for i in range(0, len(self.input_df.columns), cols_per_row)]
            
            for col_group in col_groups:
                cols = st.columns(cols_per_row)
                for i, col_name in enumerate(col_group):
                    self.categorical_attributes[col_name] = cols[i].checkbox(
                        f"{col_name} (unique values: {self.input_df[col_name].nunique()})",
                        value=detected_categorical[col_name]
                    )
            
            st.write("Unique Identifiers")
            key_cols = st.columns(cols_per_row)
            for i, col_name in enumerate(self.input_df.columns[:cols_per_row]):
                self.candidate_keys[col_name] = key_cols[i].checkbox(
                    f"{col_name} as identifier",
                    value=False
                )
    
    def forge_quantum_data(self) -> bool:
        """Generate synthetic data based on user configuration."""
        if self.input_data_file is None:
            st.warning("Please upload a dataset first.")
            return False
        
        with st.spinner("Forging quantum data..."):
            try:
                start_time = time.time()
                
                # Initialize the DataDescriber
                describer = MatrixDescriber(category_threshold=self.threshold_value)
                
                # Describe dataset based on selected mode
                if "Independent" in self.mode:
                    describer.describe_dataset_in_independent_attribute_mode(
                        dataset_file=str(self.input_data_file),
                        epsilon=self.epsilon,
                        attribute_to_is_categorical=self.categorical_attributes,
                        attribute_to_is_candidate_key=self.candidate_keys
                    )
                else:  # Correlated mode
                    describer.describe_dataset_in_correlated_attribute_mode(
                        dataset_file=str(self.input_data_file),
                        epsilon=self.epsilon,
                        k=self.degree_of_bayesian_network,
                        attribute_to_is_categorical=self.categorical_attributes,
                        attribute_to_is_candidate_key=self.candidate_keys
                    )
                
                # Save the dataset description
                describer.save_dataset_description_to_file(str(self.description_file))
                
                # Generate synthetic data
                generator = MatrixGenerator()
                
                if "Independent" in self.mode:
                    generator.generate_dataset_in_independent_mode(
                        self.num_tuples, 
                        str(self.description_file)
                    )
                else:  # Correlated mode
                    generator.generate_dataset_in_correlated_attribute_mode(
                        self.num_tuples, 
                        str(self.description_file)
                    )
                
                # Save the synthetic data
                generator.save_synthetic_data(str(self.synthetic_data_file))
                
                # Load the synthetic data
                self.synthetic_df = pd.read_csv(str(self.synthetic_data_file))
                
                # Load attribute description
                self.attribute_description = parse_json_config(str(self.description_file))['attribute_description']
                
                elapsed_time = time.time() - start_time
                st.success(f"Successfully forged {self.num_tuples} quantum records in {elapsed_time:.2f} seconds")
                
                return True
            except Exception as e:
                st.error(f"Error forging quantum data: {str(e)}")
                logger.error(f"Error forging quantum data: {str(e)}", exc_info=True)
                return False
    
    def visualize_quantum_matrix(self) -> None:
        """Display and analyze the quantum data results."""
        if self.synthetic_df is None:
            return
        
        st.header("Quantum Matrix Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Quantum Data", "Comparison", "Visualizations", "Export"])
        
        with tab1:
            st.subheader("Quantum Dataset Preview")
            st.dataframe(self.synthetic_df.head(10))
            
            st.subheader("Quantum Data Statistics")
            st.dataframe(self.synthetic_df.describe())
        
        with tab2:
            st.subheader("Source vs Quantum Data Comparison")
            
            # Compare basic statistics
            st.write("Record Count Comparison")
            count_comparison = pd.DataFrame({
                'Dataset': ['Source', 'Quantum'],
                'Records': [len(self.input_df), len(self.synthetic_df)]
            })
            st.bar_chart(count_comparison.set_index('Dataset'))
            
            # Compare distributions for selected columns
            st.write("Select attributes to compare distributions:")
            selected_columns = st.multiselect(
                "Attributes to compare",
                options=self.input_df.columns.tolist(),
                default=self.input_df.select_dtypes(include=['number']).columns.tolist()[:2]
            )
            
            if selected_columns:
                for col in selected_columns:
                    st.write(f"Distribution comparison for: {col}")
                    
                    if col in self.input_df.select_dtypes(include=['number']).columns:
                        # For numeric columns, show histograms
                        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                        
                        # Original data histogram
                        self.input_df[col].hist(ax=ax[0], bins=20)
                        ax[0].set_title(f"Source - {col}")
                        
                        # Synthetic data histogram
                        self.synthetic_df[col].hist(ax=ax[1], bins=20)
                        ax[1].set_title(f"Quantum - {col}")
                        
                        st.pyplot(fig)
                    else:
                        # For categorical columns, show bar charts
                        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
                        
                        # Original data
                        orig_counts = self.input_df[col].value_counts().sort_index()
                        orig_counts.plot(kind='bar', ax=ax[0])
                        ax[0].set_title(f"Source - {col}")
                        ax[0].tick_params(axis='x', rotation=45)
                        
                        # Synthetic data
                        synth_counts = self.synthetic_df[col].value_counts().sort_index()
                        synth_counts.plot(kind='bar', ax=ax[1])
                        ax[1].set_title(f"Quantum - {col}")
                        ax[1].tick_params(axis='x', rotation=45)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
            
            # Show mutual information heatmap if in correlated mode
            if "Correlated" in self.mode:
                st.subheader("Correlation Analysis")
                
                try:
                    inspector = PatternAnalyzer(self.input_df, self.synthetic_df, self.attribute_description)
                    
                    # Get the mutual information heatmap
                    fig = plt.figure(figsize=(10, 8))
                    inspector.mutual_information_heatmap(fig=fig)
                    st.pyplot(fig)
                    
                    st.write("This heatmap shows how well the correlations between attributes are preserved in the quantum data.")
                except Exception as e:
                    st.error(f"Error generating correlation analysis: {str(e)}")
        
        with tab3:
            st.subheader("Data Visualizations")
            
            # Allow user to select visualization type
            viz_type = st.selectbox(
                "Select Visualization Type",
                ["Pair Plot", "Correlation Heatmap", "Box Plot"]
            )
            
            numeric_cols = self.input_df.select_dtypes(include=['number']).columns.tolist()
            
            if viz_type == "Pair Plot" and len(numeric_cols) >= 2:
                # Allow user to select columns for pair plot
                pair_cols = st.multiselect(
                    "Select attributes for pair plot (2-4 recommended)",
                    options=numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols[:2]
                )
                
                if len(pair_cols) >= 2:
                    st.write("Source Data Pair Plot")
                    fig1 = sns.pairplot(self.input_df[pair_cols])
                    st.pyplot(fig1)
                    
                    st.write("Quantum Data Pair Plot")
                    fig2 = sns.pairplot(self.synthetic_df[pair_cols])
                    st.pyplot(fig2)
                else:
                    st.info("Please select at least 2 attributes for the pair plot")
            
            elif viz_type == "Correlation Heatmap":
                st.write("Source Data Correlation")
                fig1, ax1 = plt.subplots(figsize=(10, 8))
                corr_matrix = self.input_df.select_dtypes(include=['number']).corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax1)
                st.pyplot(fig1)
                
                st.write("Quantum Data Correlation")
                fig2, ax2 = plt.subplots(figsize=(10, 8))
                synth_corr = self.synthetic_df.select_dtypes(include=['number']).corr()
                sns.heatmap(synth_corr, annot=True, cmap='coolwarm', ax=ax2)
                st.pyplot(fig2)
            
            elif viz_type == "Box Plot":
                # Allow user to select column for box plot
                box_col = st.selectbox(
                    "Select attribute for box plot",
                    options=numeric_cols
                )
                
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                
                # Original data box plot
                sns.boxplot(y=self.input_df[box_col], ax=ax[0])
                ax[0].set_title(f"Source - {box_col}")
                
                # Synthetic data box plot
                sns.boxplot(y=self.synthetic_df[box_col], ax=ax[1])
                ax[1].set_title(f"Quantum - {box_col}")
                
                st.pyplot(fig)
        
        with tab4:
            st.subheader("Export Quantum Data")
            
            # Provide download options
            csv_data = self.synthetic_df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv_data,
                file_name="quantum_data.csv",
                mime="text/csv"
            )
            
            # Also provide the data description file
            with open(self.description_file, 'r') as f:
                json_data = f.read()
            
            st.download_button(
                label="Download Matrix Configuration (JSON)",
                data=json_data,
                file_name="matrix_config.json",
                mime="application/json"
            )
            
            # Option to download a sample analysis report
            if st.button("Generate Analysis Report"):
                report_html = self.create_quantum_analysis()
                st.download_button(
                    label="Download Analysis Report",
                    data=report_html,
                    file_name="quantum_data_analysis.html",
                    mime="text/html"
                )
    
    def create_quantum_analysis(self) -> str:
        """Generate an HTML report comparing original and quantum data."""
        buffer = io.StringIO()
        
        buffer.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quantum Data Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #2c3e50; }
                .container { max-width: 1200px; margin: 0 auto; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .chart-container { margin: 20px 0; }
                .footer { margin-top: 30px; font-size: 0.8em; color: #7f8c8d; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Quantum Data Analysis Report</h1>
                <p>Generated on: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
                
                <h2>Dataset Overview</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Source Dataset</th>
                        <th>Quantum Dataset</th>
                    </tr>
                    <tr>
                        <td>Number of Records</td>
                        <td>""" + str(len(self.input_df)) + """</td>
                        <td>""" + str(len(self.synthetic_df)) + """</td>
                    </tr>
                    <tr>
                        <td>Number of Attributes</td>
                        <td>""" + str(len(self.input_df.columns)) + """</td>
                        <td>""" + str(len(self.synthetic_df.columns)) + """</td>
                    </tr>
                </table>
                
                <h2>Quantum Forge Configuration</h2>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Forge Mode</td>
                        <td>""" + self.mode + """</td>
                    </tr>
                    <tr>
                        <td>Epsilon (Privacy Budget)</td>
                        <td>""" + str(self.epsilon) + """</td>
                    </tr>
                    <tr>
                        <td>Number of Records Generated</td>
                        <td>""" + str(self.num_tuples) + """</td>
                    </tr>
                </table>
                
                <h2>Attribute Statistics Comparison</h2>
        """)
        
        # Add statistics for each column
        for col in self.input_df.columns:
            buffer.write(f"<h3>Attribute: {col}</h3>")
            
            if col in self.input_df.select_dtypes(include=['number']).columns:
                # Numeric column
                orig_stats = self.input_df[col].describe()
                synth_stats = self.synthetic_df[col].describe()
                
                buffer.write("<table><tr><th>Statistic</th><th>Source</th><th>Quantum</th></tr>")
                for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                    buffer.write(f"<tr><td>{stat}</td><td>{orig_stats[stat]:.4f}</td><td>{synth_stats[stat]:.4f}</td></tr>")
                buffer.write("</table>")
                
                # Create histogram comparison
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                self.input_df[col].hist(ax=ax[0], bins=20)
                ax[0].set_title(f"Source - {col}")
                self.synthetic_df[col].hist(ax=ax[1], bins=20)
                ax[1].set_title(f"Quantum - {col}")
                
                # Save figure to base64
                canvas = FigureCanvas(fig)
                img_data = io.BytesIO()
                canvas.print_png(img_data)
                img_b64 = base64.b64encode(img_data.getvalue()).decode()
                
                buffer.write(f'<div class="chart-container"><img src="data:image/png;base64,{img_b64}" /></div>')
                plt.close(fig)
            else:
                # Categorical column
                orig_counts = self.input_df[col].value_counts().sort_index()
                synth_counts = self.synthetic_df[col].value_counts().sort_index()
                
                buffer.write("<table><tr><th>Value</th><th>Source Count</th><th>Source %</th><th>Quantum Count</th><th>Quantum %</th></tr>")
                
                # Get all unique values from both datasets
                all_values = set(orig_counts.index) | set(synth_counts.index)
                
                for val in sorted(all_values):
                    orig_count = orig_counts.get(val, 0)
                    synth_count = synth_counts.get(val, 0)
                    orig_pct = orig_count / len(self.input_df) * 100 if len(self.input_df) > 0 else 0
                    synth_pct = synth_count / len(self.synthetic_df) * 100 if len(self.synthetic_df) > 0 else 0
                    
                    buffer.write(f"<tr><td>{val}</td><td>{orig_count}</td><td>{orig_pct:.2f}%</td><td>{synth_count}</td><td>{synth_pct:.2f}%</td></tr>")
                
                buffer.write("</table>")
        
        buffer.write("""
                <div class="footer">
                    <p>Generated using Quantum Forge Engine by Jasma Team - git@github.com:jasmadata/jasma-data-synthesizer-p1-v1.git</p>
                </div>
            </div>
        </body>
        </html>
        """)
        
        return buffer.getvalue()
    
    def run(self) -> None:
        """Run the Streamlit application."""
        self.configure_control_panel()
        
        if self.ingest_source_data():
            if st.button("Forge Quantum Data", type="primary"):
                if self.forge_quantum_data():
                    self.visualize_quantum_matrix()

if __name__ == "__main__":
    app = QuantumForgeEngine()
    app.run()

