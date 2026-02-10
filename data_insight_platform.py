"""
PreMortemML - Pre-Training Risk Analysis & Quality Assessment

Before you train anything, understand what could go wrong.
Built by amukta14
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from premortemml import Datalab
import matplotlib.pyplot as plt
from io import StringIO
import sys
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="PreMortemML | amukta14",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# Modern, clean CSS
st.markdown("""
<style>
    /* @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'); */
    
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0a0a0a;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .author-link {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .author-link a {
        color: #2563eb;
        text-decoration: none;
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    .author-link a:hover {
        text-decoration: underline;
    }
    
    .info-box {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 3px solid #2563eb;
        margin: 1.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-top: 3px solid #2563eb;
    }
    
    .warning-box {
        background: #fef3c7;
        padding: 1rem;
        border-radius: 6px;
        border-left: 3px solid #f59e0b;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d1fae5;
        padding: 1rem;
        border-radius: 6px;
        border-left: 3px solid #10b981;
        margin: 1rem 0;
    }
    
    .stButton>button {
        background: #2563eb;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        border: none;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background: #1d4ed8;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    h1, h2, h3 {
        color: #0a0a0a;
        font-weight: 600;
    }
    
    .sidebar .sidebar-content {
        background: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Header
st.markdown('<p class="main-header">PreMortemML</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Pre-Training Risk Analysis & Quality Assessment</p>', unsafe_allow_html=True)
st.markdown('<div class="author-link">Built by <a href="https://github.com/amukta14" target="_blank">amukta14</a></div>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Navigation",
    ["Quality Analysis", "Settings"],
    label_visibility="collapsed",
)

# ==================== HELPER FUNCTIONS ====================

@st.cache_data
def preprocess_data(df, feature_columns, label_column, handle_missing='drop', scale_features=True):
    """Preprocess data: handle missing values, encode labels, scale features"""
    df_clean = df.copy()
    
    if handle_missing == 'drop':
        df_clean = df_clean.dropna(subset=feature_columns + [label_column])
    elif handle_missing == 'fill_mean':
        df_clean[feature_columns] = df_clean[feature_columns].fillna(df_clean[feature_columns].mean())
        df_clean[label_column] = df_clean[label_column].fillna(df_clean[label_column].mode()[0] if len(df_clean[label_column].mode()) > 0 else 0)
    
    # Robust feature handling:
    # - Try to coerce selected feature columns to numeric
    # - Drop columns that are completely non-numeric (all NaN after coercion)
    # - Warn the user instead of crashing on bad data
    numeric_feature_columns = []
    dropped_feature_columns = []
    for col in feature_columns:
        # Try to convert to numeric, marking non-convertible values as NaN
        coerced = pd.to_numeric(df_clean[col], errors='coerce')
        if coerced.notna().sum() == 0:
            # Column is effectively non-numeric
            dropped_feature_columns.append(col)
        else:
            df_clean[col] = coerced
            numeric_feature_columns.append(col)

    if dropped_feature_columns:
        st.warning(
            "Ignoring non-numeric feature columns that could not be converted: "
            + ", ".join(dropped_feature_columns)
        )

    if len(numeric_feature_columns) == 0:
        raise ValueError(
            "No usable numeric feature columns found after cleaning. "
            "Please select at least one numeric feature column (e.g. counts, scores, or numeric IDs) "
            "and avoid purely text columns like 'Dollars (millions)'."
        )

    X = df_clean[numeric_feature_columns].values.astype(float)
    y_raw = df_clean[label_column].values
    
    if y_raw.dtype == 'object' or isinstance(y_raw[0], str):
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    else:
        y = y_raw.astype(int)
        label_mapping = None
    
    if scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, y, df_clean, label_mapping

def get_optimal_cv_folds(y, max_folds=5):
    """Dynamically determine optimal CV folds based on class distribution"""
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_size = min(class_counts)
    
    if min_class_size < 2:
        return 0, min_class_size, unique_classes
    
    optimal_folds = min(max_folds, min_class_size, len(y) // 2)
    if optimal_folds < 2:
        optimal_folds = 0
    
    return optimal_folds, min_class_size, unique_classes

def get_predicted_probabilities(X, y, cv_folds, model_type='logistic', use_sampling=False, sample_size=None):
    """Get predicted probabilities with smart handling for any dataset size"""
    n_samples = len(y)
    
    if use_sampling and sample_size and n_samples > sample_size:
        indices = np.random.choice(n_samples, size=sample_size, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
        st.info(f"Using random sample of {sample_size:,} rows from {n_samples:,} total rows")
    else:
        X_sample = X
        y_sample = y
    
    if model_type == 'logistic':
        model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    elif model_type == 'random_forest':
        n_estimators = min(100, max(10, len(X_sample) // 10))
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1, max_depth=10)
    else:
        model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    
    optimal_folds, min_class_size, unique_classes = get_optimal_cv_folds(y_sample, cv_folds)
    
    if min_class_size < 2:
        st.warning(f"Some classes have only {min_class_size} sample(s). Using simple model fit.")
        model.fit(X_sample, y_sample)
        pred_probs = model.predict_proba(X_sample)
        return (pred_probs, 0) if not (use_sampling and sample_size and n_samples > sample_size) else (model.predict_proba(X), 0)
    
    if optimal_folds >= 2:
        try:
            skf = StratifiedKFold(n_splits=optimal_folds, shuffle=True, random_state=42)
            pred_probs = cross_val_predict(
                estimator=model,
                X=X_sample,
                y=y_sample,
                cv=skf,
                method="predict_proba",
                n_jobs=-1
            )
        except ValueError as e:
            st.warning(f"Cross-validation failed: {str(e)}. Using simple model fit.")
            model.fit(X_sample, y_sample)
            pred_probs = model.predict_proba(X_sample)
            optimal_folds = 0
    else:
        if optimal_folds == 0:
            st.info(f"Dataset too small for cross-validation. Using simple model fit.")
        model.fit(X_sample, y_sample)
        pred_probs = model.predict_proba(X_sample)
    
    if use_sampling and sample_size and n_samples > sample_size:
        model.fit(X_sample, y_sample)
        pred_probs_full = model.predict_proba(X)
        return pred_probs_full, optimal_folds
    else:
        return pred_probs, optimal_folds

def run_quality_analysis(df, X, y, label_column, issue_types=None, use_features=True):
    """Run quality analysis"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Training model and getting predictions...")
    progress_bar.progress(20)
    
    cv_folds = st.session_state.get('cv_folds', 5)
    model_type = st.session_state.get('model_type', 'logistic')
    use_sampling = st.session_state.get('use_sampling', False)
    sample_size = st.session_state.get('sample_size', 10000)
    
    pred_probs, actual_cv_folds = get_predicted_probabilities(
        X, y, cv_folds, model_type, use_sampling, sample_size
    )
    
    progress_bar.progress(50)
    status_text.text("Initializing analysis engine...")
    progress_bar.progress(60)
    
    lab = Datalab(data=df, label_name=label_column)
    
    status_text.text("Analyzing data for issues...")
    progress_bar.progress(70)
    
    features_to_use = X if use_features else None
    
    if issue_types:
        lab.find_issues(features=features_to_use, pred_probs=pred_probs, issue_types=issue_types)
    else:
        lab.find_issues(features=features_to_use, pred_probs=pred_probs)
    
    progress_bar.progress(100)
    status_text.text("Analysis complete")
    progress_bar.empty()
    status_text.empty()
    
    return lab, pred_probs


# ==================== DISPLAY FUNCTIONS ====================

def display_quality_results(lab):
    """Display quality analysis results"""
    issues = lab.get_issues()
    
    st.header("Quality Analysis Results")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_issues = 0
    issue_counts = {}
    for issue_type in ['label', 'outlier', 'near_duplicate', 'duplicate', 'class_imbalance']:
        issue_col = f'is_{issue_type}_issue'
        if issue_col in issues.columns:
            count = int(issues[issue_col].sum())
            issue_counts[issue_type] = count
            total_issues += count
    
    with col1:
        st.metric("Total Issues", f"{total_issues:,}")
    with col2:
        st.metric("Label Issues", f"{issue_counts.get('label', 0):,}")
    with col3:
        st.metric("Outliers", f"{issue_counts.get('outlier', 0):,}")
    with col4:
        st.metric("Duplicates", f"{issue_counts.get('near_duplicate', 0):,}")
    with col5:
        st.metric("Imbalance", f"{issue_counts.get('class_imbalance', 0):,}")
    
    st.subheader("Detailed Report")
    with st.expander("View Full Report", expanded=False):
        old_stdout = sys.stdout
        sys.stdout = report_output = StringIO()
        lab.report()
        sys.stdout = old_stdout
        st.text(report_output.getvalue())
    
    st.subheader("Issue Details")
    tabs = ["Overview"]
    if 'is_label_issue' in issues.columns:
        tabs.append("Label Issues")
    if 'is_outlier_issue' in issues.columns:
        tabs.append("Outliers")
    if 'is_near_duplicate_issue' in issues.columns:
        tabs.append("Duplicates")
    
    tab_objects = st.tabs(tabs)
    
    with tab_objects[0]:
        st.dataframe(issues.head(100))
    
    tab_idx = 1
    if 'is_label_issue' in issues.columns:
        with tab_objects[tab_idx]:
            label_issues_df = issues[issues['is_label_issue']]
            if len(label_issues_df) > 0:
                st.dataframe(label_issues_df[['is_label_issue', 'label_score']].head(100))
            else:
                st.info("No label issues detected")
        tab_idx += 1
    
    if 'is_outlier_issue' in issues.columns:
        with tab_objects[tab_idx]:
            outlier_issues_df = issues[issues['is_outlier_issue']]
            if len(outlier_issues_df) > 0:
                st.dataframe(outlier_issues_df[['is_outlier_issue', 'outlier_score']].head(100))
            else:
                st.info("No outliers detected")
        tab_idx += 1
    
    if 'is_near_duplicate_issue' in issues.columns:
        with tab_objects[tab_idx]:
            dup_issues_df = issues[issues['is_near_duplicate_issue']]
            if len(dup_issues_df) > 0:
                st.dataframe(dup_issues_df[['is_near_duplicate_issue', 'near_duplicate_score']].head(100))
            else:
                st.info("No duplicates detected")
    
    csv = issues.to_csv(index=False)
    st.download_button(
        label="Download Issues CSV",
        data=csv,
        file_name="quality_analysis_results.csv",
        mime="text/csv"
    )

# ==================== MAIN PAGES ====================

if page == "Quality Analysis":
    st.header("Data Quality Analysis")
    
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            with st.expander("Preview Data"):
                st.dataframe(df.head(20))
            
            st.subheader("Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                label_column = st.selectbox("Label Column", options=df.columns.tolist())
            
            with col2:
                all_cols = [col for col in df.columns if col != label_column]
                feature_columns = st.multiselect(
                    "Feature Columns (empty = auto-detect numeric)",
                    options=all_cols
                )
            
            if len(feature_columns) == 0:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                feature_columns = [col for col in numeric_cols if col != label_column]
                if len(feature_columns) > 0:
                    st.info(f"Auto-detected {len(feature_columns)} numeric feature columns")
            
            if st.button("Run Quality Analysis", type="primary"):
                try:
                    handle_missing = st.session_state.get('handle_missing', 'drop')
                    scale_features = st.session_state.get('scale_features', True)
                    
                    X, y, df_clean, label_mapping = preprocess_data(
                        df, feature_columns, label_column, handle_missing, scale_features
                    )
                    
                    if len(df_clean) == 0:
                        st.error("No data remaining after preprocessing")
                        st.stop()
                    
                    unique_classes = np.unique(y)
                    if len(unique_classes) < 2:
                        st.error(f"Need at least 2 classes. Found: {unique_classes}")
                        st.stop()
                    
                    issue_types = {}
                    if st.session_state.get('detect_label_issues', True):
                        issue_types['label'] = {}
                    if st.session_state.get('detect_outliers', True):
                        issue_types['outlier'] = {}
                    if st.session_state.get('detect_duplicates', True):
                        issue_types['near_duplicate'] = {}
                    
                    issue_types = issue_types if issue_types else None
                    use_features = st.session_state.get('detect_outliers', True) or st.session_state.get('detect_duplicates', True)
                    
                    lab, pred_probs = run_quality_analysis(
                        df_clean, X, y, label_column, issue_types, use_features
                    )
                    
                    st.session_state.analysis_results = {
                        'lab': lab,
                        'df': df_clean,
                        'X': X,
                        'y': y,
                        'pred_probs': pred_probs,
                        'label_mapping': label_mapping
                    }
                    
                    display_quality_results(lab)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    with st.expander("Error Details"):
                        st.exception(e)
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

elif page == "Settings":
    st.header("Analysis Settings")
    
    st.subheader("Model Configuration")
    st.session_state['model_type'] = st.selectbox(
        "Model Type",
        options=['logistic', 'random_forest'],
        index=0
    )
    
    st.session_state['cv_folds'] = st.slider(
        "Cross-Validation Folds",
        min_value=2,
        max_value=10,
        value=5
    )
    
    st.subheader("Performance")
    st.session_state['use_sampling'] = st.checkbox("Use Sampling for Large Datasets", value=False)
    
    if st.session_state['use_sampling']:
        st.session_state['sample_size'] = st.number_input(
            "Sample Size",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000
        )
    
    st.subheader("Preprocessing")
    st.session_state['handle_missing'] = st.selectbox(
        "Handle Missing Values",
        options=['drop', 'fill_mean'],
        index=0
    )
    
    st.session_state['scale_features'] = st.checkbox("Scale Features", value=True)
    
    st.subheader("Issue Detection")
    st.session_state['detect_label_issues'] = st.checkbox("Detect Label Issues", value=True)
    st.session_state['detect_outliers'] = st.checkbox("Detect Outliers", value=True)
    st.session_state['detect_duplicates'] = st.checkbox("Detect Duplicates", value=True)
    
    st.success("Settings saved")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>Built by <a href="https://github.com/amukta14" style="color: #2563eb; text-decoration: none;">amukta14</a></p>
    <p>PreMortemML</p>
</div>
""", unsafe_allow_html=True)
