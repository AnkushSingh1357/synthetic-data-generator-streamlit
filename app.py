import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import io
import warnings

# Suppress warnings for a cleaner interface
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Synthetic Data Generation with CTGAN",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---

@st.cache_data
def load_data(uploaded_file):
    """Loads data from an uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode('utf-8')

def get_column_types(df):
    """Identifies numeric and categorical columns in a DataFrame."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numeric_cols, categorical_cols

@st.cache_resource
def train_ctgan_model(_df, epochs=300):
    """Trains the CTGAN model and returns the synthesizer object."""
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=_df)
    
    # FIX: The `progress_callback` argument is deprecated in newer sdv versions.
    # The `verbose=True` argument will print progress to the console where Streamlit is running.
    synthesizer = CTGANSynthesizer(metadata, epochs=epochs, verbose=True)
    
    synthesizer.fit(_df)
    
    return synthesizer

def evaluate_model(model, X_test, y_test):
    """Calculates evaluation metrics for a given model."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1}

# --- Main Application ---

st.title("🤖 Synthetic Data Generation & Comparison")
st.markdown("""
Welcome to the Synthetic Data Generation tool! This application uses the **CTGAN (Conditional Tabular Generative Adversarial Network)** model from the [Synthetic Data Vault (SDV)](https://sdv.dev/) library.

**How to use this app:**
1.  **Upload your data:** Use the sidebar to upload a CSV file.
2.  **Configure settings:** Adjust the generation and ML settings in the sidebar.
3.  **Generate data:** Click the "Generate Synthetic Data" button.
4.  **Explore the results:** Use the tabs to preview data, compare distributions, and evaluate machine learning performance.
""")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("1. Upload your CSV data", type=["csv"])

    if uploaded_file:
        real_data = load_data(uploaded_file)
        
        st.subheader("2. Generation Settings")
        
        num_rows_to_generate = st.number_input(
            "Number of synthetic rows to generate",
            min_value=100,
            max_value=50000,
            value=1000,
            step=100,
            help="Specify how many rows of synthetic data you want to create."
        )
        
        epochs = st.slider(
            "Training Epochs",
            min_value=50,
            max_value=1000,
            value=300,
            step=50,
            help="Number of training cycles for the GAN. More epochs can lead to better quality but take longer."
        )

        st.subheader("3. ML Model Settings")
        if real_data is not None:
            # Drop CustomerID as it's a unique identifier and not useful for prediction
            ml_columns = [col for col in real_data.columns if col.lower() != 'customerid']
            
            # Ensure there are columns to select from
            if ml_columns:
                # Try to find a common target name, otherwise default to the last column
                default_target_index = 0
                common_targets = ['churn', 'target', 'label', 'outcome', 'class']
                for i, col in enumerate(ml_columns):
                    if col.lower() in common_targets:
                        default_target_index = i
                        break
                else: # if no common target is found, default to the last column
                    default_target_index = len(ml_columns) - 1

                target_column = st.selectbox(
                    "Select the target column for classification",
                    options=ml_columns,
                    index=default_target_index
                )
            else:
                st.warning("No suitable columns found for ML prediction after excluding 'CustomerID'.")
                target_column = None


# --- Main Panel ---
if uploaded_file is None:
    st.info("Please upload a CSV file using the sidebar to get started.")
else:
    # Initialize session state for storing synthetic data
    if 'synthetic_data' not in st.session_state:
        st.session_state.synthetic_data = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📊 Data Preview & Generation", "📈 Distribution Plots", "🤖 ML Comparison"])

    with tab1:
        st.header("Real Data Preview")
        st.write("Here are the first 5 rows of your uploaded dataset:")
        st.dataframe(real_data.head())

        numeric_cols, categorical_cols = get_column_types(real_data)
        st.write("**Detected Column Types:**")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Numeric Columns ({len(numeric_cols)}):")
            st.json(numeric_cols)
        with col2:
            st.success(f"Categorical Columns ({len(categorical_cols)}):")
            st.json(categorical_cols)

        st.divider()

        st.header("Synthetic Data Generation")
        if st.button("🚀 Generate Synthetic Data", type="primary"):
            with st.spinner(f"Training the CTGAN model for {epochs} epochs. This may take a few minutes..."):
                try:
                    # Exclude CustomerID from training data as it's just an identifier
                    training_data = real_data.drop(columns=['CustomerID'], errors='ignore')
                    synthesizer = train_ctgan_model(training_data, epochs)
                    st.session_state.synthetic_data = synthesizer.sample(num_rows=num_rows_to_generate)
                    st.session_state.model_trained = True
                    st.success("Synthetic data generated successfully!")
                except Exception as e:
                    st.error(f"An error occurred during model training: {e}")
        
        if st.session_state.model_trained:
            st.write("Here are the first 5 rows of the generated synthetic dataset:")
            st.dataframe(st.session_state.synthetic_data.head())
            
            # Download button
            csv_data = convert_df_to_csv(st.session_state.synthetic_data)
            st.download_button(
                label="📥 Download Synthetic Data as CSV",
                data=csv_data,
                file_name=f"synthetic_data_{uploaded_file.name}",
                mime="text/csv",
            )

    with tab2:
        st.header("Comparison of Data Distributions")
        if not st.session_state.model_trained:
            st.warning("Please generate the synthetic data in the 'Data Preview & Generation' tab first.")
        else:
            st.info("Showing overlapping density plots for all **numeric** columns.")
            synthetic_data = st.session_state.synthetic_data
            
            # Filter to numeric columns that exist in both dataframes
            real_numeric = real_data.select_dtypes(include=np.number).columns
            synth_numeric = synthetic_data.select_dtypes(include=np.number).columns
            plottable_numeric_cols = list(set(real_numeric) & set(synth_numeric))

            if not plottable_numeric_cols:
                st.warning("No common numeric columns found to plot.")
            else:
                # Determine grid layout
                num_plots = len(plottable_numeric_cols)
                cols_per_row = 3
                num_rows = (num_plots + cols_per_row - 1) // cols_per_row
                
                for i in range(num_rows):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        idx = i * cols_per_row + j
                        if idx < num_plots:
                            col_name = plottable_numeric_cols[idx]
                            with cols[j]:
                                fig, ax = plt.subplots()
                                sns.kdeplot(real_data[col_name], ax=ax, label='Real Data', fill=True, color='blue', alpha=0.6)
                                sns.kdeplot(synthetic_data[col_name], ax=ax, label='Synthetic Data', fill=True, color='orange', alpha=0.6)
                                ax.set_title(f'Distribution for {col_name}', fontsize=10)
                                ax.legend()
                                plt.tight_layout()
                                st.pyplot(fig)

    with tab3:
        st.header("Machine Learning Model Performance Comparison")
        if not st.session_state.model_trained:
            st.warning("Please generate the synthetic data in the 'Data Preview & Generation' tab first.")
        elif target_column is None:
            st.warning("Please select a target column in the sidebar to run the ML comparison.")
        else:
            st.info(f"Training a **Decision Tree Classifier** to predict the target column: **'{target_column}'**.")
            
            try:
                # --- Data Preprocessing for ML ---
                real_ml_data = real_data.drop(columns=['CustomerID'], errors='ignore')

                # Combine real and synthetic data for consistent encoding
                combined_data = pd.concat([real_ml_data, st.session_state.synthetic_data], ignore_index=True)
                
                # Use LabelEncoder for all categorical columns
                encoders = {}
                for col in combined_data.select_dtypes(include=['object', 'category']).columns:
                    le = LabelEncoder()
                    combined_data[col] = le.fit_transform(combined_data[col].astype(str))
                    encoders[col] = le

                # Separate back into real and synthetic
                processed_real = combined_data.iloc[:len(real_ml_data)]
                processed_synthetic = combined_data.iloc[len(real_ml_data):]
                
                # --- Real Data Model ---
                st.subheader("1. Model Trained on Real Data")
                X_real = processed_real.drop(columns=[target_column])
                y_real = processed_real[target_column]
                X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
                    X_real, y_real, test_size=0.3, random_state=42, stratify=y_real
                )
                
                model_real = DecisionTreeClassifier(random_state=42)
                model_real.fit(X_train_real, y_train_real)
                
                metrics_real = evaluate_model(model_real, X_test_real, y_test_real)
                st.write("Performance on the **real test set**:")
                st.json({k: f"{v:.4f}" for k, v in metrics_real.items()})

                # --- Synthetic Data Model ---
                st.subheader("2. Model Trained on Synthetic Data")
                X_synthetic = processed_synthetic.drop(columns=[target_column])
                y_synthetic = processed_synthetic[target_column]
                
                model_synthetic = DecisionTreeClassifier(random_state=42)
                model_synthetic.fit(X_synthetic, y_synthetic)
                
                metrics_synthetic = evaluate_model(model_synthetic, X_test_real, y_test_real)
                st.write("Performance on the **real test set**:")
                st.json({k: f"{v:.4f}" for k, v in metrics_synthetic.items()})

                # --- Performance Comparison Plot ---
                st.subheader("3. Performance Comparison")
                metrics_df = pd.DataFrame({
                    'Real Data Model': metrics_real,
                    'Synthetic Data Model': metrics_synthetic
                }).T.reset_index()
                metrics_df = metrics_df.rename(columns={'index': 'Model'})
                
                fig, ax = plt.subplots(figsize=(10, 6))
                metrics_df.set_index('Model').plot(kind='bar', ax=ax, rot=0)
                ax.set_title('Comparison of ML Model Performance')
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1.1)
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.3f')
                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"An error occurred during ML model evaluation: {e}")
                st.warning("This can happen if the target column has too few samples for stratification or if data types are inconsistent. Please check your data and target column selection.")
