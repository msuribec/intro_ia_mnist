import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Drawing canvas
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST Classifier Studio",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-left: 4px solid #667eea;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.4rem 0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e293b;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        color: #78350f;
        font-size: 0.9rem;
    }
    .success-box {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        color: #065f46;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS / CACHE
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target
    return X, y, digits

def data_quality_report(X, y):
    report = {}
    report['n_samples'] = X.shape[0]
    report['n_features'] = X.shape[1]
    report['n_classes'] = len(np.unique(y))
    report['missing_values'] = int(np.isnan(X).sum())
    report['duplicate_rows'] = int(pd.DataFrame(X).duplicated().sum())
    report['class_counts'] = pd.Series(y).value_counts().sort_index()
    report['feature_range'] = (float(X.min()), float(X.max()))
    report['zero_variance_feats'] = int((X.var(axis=0) == 0).sum())
    return report

def build_pipeline(model, use_pca, pca_components=30):
    steps = [('scaler', StandardScaler())]
    if use_pca:
        steps.append(('pca', PCA(n_components=pca_components, random_state=42)))
    steps.append(('clf', model))
    return Pipeline(steps)

MODELS = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'ANN (MLP)': MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                               random_state=42, early_stopping=True),
}

MODEL_COLORS = {
    'KNN': '#3b82f6',
    'Naive Bayes': '#f59e0b',
    'SVM': '#8b5cf6',
    'Random Forest': '#10b981',
    'Decision Tree': '#ef4444',
    'Logistic Regression': '#06b6d4',
    'ANN (MLP)': '#ec4899',
}

@st.cache_data(show_spinner=False)
def run_experiments(train_size, use_pca, pca_components, cv_folds):
    X, y, _ = load_data()

    # Split BEFORE any fitting — no leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42, stratify=y
    )

    results = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, base_model in MODELS.items():
        pipe = build_pipeline(base_model, use_pca, pca_components)

        # Cross-validation on TRAINING data only (no leakage)
        cv_res = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring=['accuracy', 'f1_macro'],
            return_train_score=True, n_jobs=-1
        )

        # Fit on full training set, predict on held-out test
        pipe.fit(X_train, y_train)
        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)

        results[name] = {
            'pipe': pipe,
            'cv_train_acc': cv_res['train_accuracy'],
            'cv_val_acc': cv_res['test_accuracy'],
            'cv_train_f1': cv_res['train_f1_macro'],
            'cv_val_f1': cv_res['test_f1_macro'],
            'train_acc': accuracy_score(y_train, y_pred_train),
            'test_acc': accuracy_score(y_test, y_pred_test),
            'y_pred_test': y_pred_test,
            'y_test': y_test,
        }

    return results, X_train, X_test, y_train, y_test


def preprocess_canvas_image(image_data):
    """Convert canvas RGBA → 8x8 grayscale matching sklearn digits format."""
    img = Image.fromarray(image_data.astype(np.uint8))
    img = img.convert('L')
    img_array = np.array(img)
    # Invert if white-on-black
    if img_array.mean() > 127:
        img_array = 255 - img_array
    img_resized = cv2.resize(img_array, (8, 8), interpolation=cv2.INTER_AREA)
    img_scaled = img_resized / img_resized.max() * 16.0 if img_resized.max() > 0 else img_resized
    return img_scaled.flatten().reshape(1, -1)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    train_pct = st.slider(
        "Training data size (%)",
        min_value=30, max_value=90, value=70, step=5,
        help="Percentage of data used for training. The rest becomes the test set."
    )
    train_size = train_pct / 100.0

    st.markdown("---")
    use_pca = st.checkbox("Apply PCA preprocessing", value=False,
                          help="Reduces feature dimensions before training.")
    pca_components = 30
    if use_pca:
        pca_components = st.slider("PCA components", 10, 60, 30, 5)

    st.markdown("---")
    cv_folds = st.slider("Cross-validation folds (StratifiedKFold)",
                         min_value=3, max_value=10, value=5, step=1)

    st.markdown("---")
    run_btn = st.button("🚀 Train All Models", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#94a3b8;'>
    ✅ No data leakage: scaling & PCA fit only on training data via Pipeline.<br><br>
    📊 Cross-validation performed on train split only.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">🔢 MNIST Classifier Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Train, evaluate, and interact with 7 classification models on handwritten digits</div>', unsafe_allow_html=True)

tab_data, tab_train, tab_perf, tab_draw = st.tabs([
    "📊 Data Quality", "🏋️ Training", "📈 Performance", "✏️ Draw & Classify"
])

# ═══════════════════════════════════════════════
# TAB 1 — DATA QUALITY
# ═══════════════════════════════════════════════
with tab_data:
    X_raw, y_raw, digits_obj = load_data()
    report = data_quality_report(X_raw, y_raw)

    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{report['n_samples']:,}</div>
            <div class="metric-label">Total Samples</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{report['n_features']}</div>
            <div class="metric-label">Features (8×8 pixels)</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{report['n_classes']}</div>
            <div class="metric-label">Classes (0–9)</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{report['missing_values']}</div>
            <div class="metric-label">Missing Values</div></div>""", unsafe_allow_html=True)

    st.markdown("")

    if report['missing_values'] == 0 and report['zero_variance_feats'] == 0:
        st.markdown('<div class="success-box">✅ No missing values or zero-variance features detected. Dataset is clean.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warning-box">⚠️ Found {report["missing_values"]} missing values and {report["zero_variance_feats"]} zero-variance features.</div>',
                    unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-title">Class Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(report['class_counts'].index, report['class_counts'].values,
                      color=[MODEL_COLORS.get(k, '#667eea') for k in ['KNN','Naive Bayes','SVM','Random Forest',
                                                                        'Decision Tree','Logistic Regression','ANN (MLP)','KNN','Naive Bayes','SVM']],
                      edgecolor='white', linewidth=0.8)
        ax.set_xlabel('Digit Class', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Samples per Class', fontsize=12, fontweight='bold')
        ax.set_xticks(range(10))
        for bar, val in zip(bars, report['class_counts'].values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(val), ha='center', va='bottom', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown('<div class="section-title">Sample Digits</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(2, 10, figsize=(10, 2.2))
        for digit in range(10):
            idx = np.where(y_raw == digit)[0]
            for row in range(2):
                ax = axes[row, digit]
                ax.imshow(X_raw[idx[row]].reshape(8, 8), cmap='Blues', vmin=0, vmax=16)
                ax.axis('off')
                if row == 0:
                    ax.set_title(str(digit), fontsize=9, fontweight='bold')
        fig.suptitle('Two samples per digit', fontsize=10, y=1.02)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-title">Pixel Intensity Distribution</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

    axes[0].hist(X_raw.flatten(), bins=17, range=(-0.5, 16.5),
                 color='#667eea', edgecolor='white', alpha=0.85)
    axes[0].set_xlabel('Pixel Value (0–16)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Global Pixel Intensity Distribution', fontsize=12, fontweight='bold')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    feature_means = X_raw.mean(axis=0).reshape(8, 8)
    im = axes[1].imshow(feature_means, cmap='YlOrRd')
    axes[1].set_title('Average Pixel Intensity per Position', fontsize=12, fontweight='bold')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im, ax=axes[1])

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    if use_pca:
        st.markdown('<div class="section-title">PCA Variance Analysis (train data preview)</div>', unsafe_allow_html=True)
        X_train_prev, _, _, _ = train_test_split(X_raw, y_raw, train_size=train_size,
                                                  random_state=42, stratify=y_raw)
        scaler_prev = StandardScaler()
        X_scaled_prev = scaler_prev.fit_transform(X_train_prev)
        pca_prev = PCA(n_components=min(60, X_scaled_prev.shape[1]), random_state=42)
        pca_prev.fit(X_scaled_prev)
        cumvar = np.cumsum(pca_prev.explained_variance_ratio_) * 100

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(range(1, len(cumvar)+1), cumvar, color='#667eea', linewidth=2)
        ax.axhline(y=95, color='#ef4444', linestyle='--', linewidth=1.2, label='95% variance')
        ax.axvline(x=pca_components, color='#10b981', linestyle='--', linewidth=1.2,
                   label=f'Selected: {pca_components} components')
        ax.fill_between(range(1, len(cumvar)+1), cumvar, alpha=0.15, color='#667eea')
        ax.set_xlabel('Number of Components', fontsize=11)
        ax.set_ylabel('Cumulative Explained Variance (%)', fontsize=11)
        ax.set_title('PCA Explained Variance Curve (fitted on training data only)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

# ═══════════════════════════════════════════════
# TAB 2 — TRAINING
# ═══════════════════════════════════════════════
with tab_train:
    st.markdown('<div class="section-title">Training Configuration</div>', unsafe_allow_html=True)

    conf_cols = st.columns(4)
    with conf_cols[0]:
        st.info(f"**Train size:** {train_pct}%")
    with conf_cols[1]:
        st.info(f"**Test size:** {100 - train_pct}%")
    with conf_cols[2]:
        st.info(f"**PCA:** {'Yes – ' + str(pca_components) + ' components' if use_pca else 'No'}")
    with conf_cols[3]:
        st.info(f"**CV folds:** {cv_folds} (Stratified K-Fold)")

    st.markdown("""
    <div class="warning-box">
    🛡️ <b>Data Leakage Prevention:</b> All preprocessing (StandardScaler + optional PCA) is wrapped inside 
    a <code>sklearn.Pipeline</code> and fitted only on the training fold during cross-validation, 
    and only on the full training set for final evaluation. The test set is never seen during fitting.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    if 'results' not in st.session_state or run_btn:
        with st.spinner("Training 7 models with cross-validation... This may take ~30 seconds."):
            results, X_tr, X_te, y_tr, y_te = run_experiments(
                train_size, use_pca, pca_components, cv_folds
            )
        st.session_state['results'] = results
        st.session_state['splits'] = (X_tr, X_te, y_tr, y_te)
        st.success("✅ All models trained successfully!")

    if 'results' in st.session_state:
        results = st.session_state['results']
        X_tr, X_te, y_tr, y_te = st.session_state['splits']

        st.markdown('<div class="section-title">Cross-Validation Results (on Training Data Only)</div>', unsafe_allow_html=True)

        cv_df = pd.DataFrame({
            'Model': list(results.keys()),
            'CV Val Acc (mean)': [results[m]['cv_val_acc'].mean() for m in results],
            'CV Val Acc (std)': [results[m]['cv_val_acc'].std() for m in results],
            'CV Train Acc (mean)': [results[m]['cv_train_acc'].mean() for m in results],
            'CV Val F1 (mean)': [results[m]['cv_val_f1'].mean() for m in results],
            'Final Test Acc': [results[m]['test_acc'] for m in results],
        }).sort_values('CV Val Acc (mean)', ascending=False).reset_index(drop=True)

        cv_df_display = cv_df.copy()
        for col in cv_df_display.columns[1:]:
            cv_df_display[col] = cv_df_display[col].map(lambda x: f"{x*100:.2f}%")

        st.dataframe(cv_df_display, use_container_width=True,
                     column_config={"Model": st.column_config.TextColumn(width="medium")})

        # Bias-Variance per model
        st.markdown('<div class="section-title">Bias–Variance per Fold (CV on Train Split)</div>', unsafe_allow_html=True)

        selected_model = st.selectbox("Select model to inspect CV folds:", list(results.keys()))
        m_res = results[selected_model]

        fig, ax = plt.subplots(figsize=(9, 4))
        folds = np.arange(1, cv_folds+1)
        ax.plot(folds, m_res['cv_train_acc']*100, 'o-', color='#10b981',
                linewidth=2, markersize=6, label='Train Accuracy (CV fold)')
        ax.plot(folds, m_res['cv_val_acc']*100, 's-', color='#667eea',
                linewidth=2, markersize=6, label='Validation Accuracy (CV fold)')
        ax.fill_between(folds, m_res['cv_train_acc']*100, m_res['cv_val_acc']*100,
                        alpha=0.12, color='#f59e0b', label='Overfit gap')
        ax.axhline(y=m_res['test_acc']*100, color='#ef4444', linestyle='--',
                   linewidth=1.5, label=f'Final Test Acc: {m_res["test_acc"]*100:.1f}%')
        ax.set_xlabel('CV Fold', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(f'{selected_model} — Bias-Variance Across CV Folds', fontsize=12, fontweight='bold')
        ax.set_xticks(folds)
        ax.legend(fontsize=10)
        ax.set_ylim(50, 105)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        st.info("👈 Press **Train All Models** in the sidebar to start.")

# ═══════════════════════════════════════════════
# TAB 3 — PERFORMANCE
# ═══════════════════════════════════════════════
with tab_perf:
    if 'results' not in st.session_state:
        st.info("👈 Press **Train All Models** in the sidebar first.")
    else:
        results = st.session_state['results']
        X_tr, X_te, y_tr, y_te = st.session_state['splits']

        model_names = list(results.keys())
        train_accs = [results[m]['train_acc']*100 for m in model_names]
        test_accs = [results[m]['test_acc']*100 for m in model_names]
        cv_val_means = [results[m]['cv_val_acc'].mean()*100 for m in model_names]
        cv_val_stds = [results[m]['cv_val_acc'].std()*100 for m in model_names]

        # ── Chart 1: Train vs Test Accuracy
        st.markdown('<div class="section-title">Train vs Test Accuracy Comparison</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 4.5))
        x = np.arange(len(model_names))
        w = 0.3
        bars1 = ax.bar(x - w/2, train_accs, w, label='Train Accuracy', color='#10b981', edgecolor='white')
        bars2 = ax.bar(x + w/2, test_accs, w, label='Test Accuracy', color='#667eea', edgecolor='white')

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title('Training vs Test Set Accuracy per Model', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.set_ylim(0, 115)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Chart 2: CV Validation Accuracy with error bars
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-title">CV Validation Accuracy (± std)</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = [MODEL_COLORS[m] for m in model_names]
            sorted_idx = np.argsort(cv_val_means)[::-1]
            sorted_names = [model_names[i] for i in sorted_idx]
            sorted_means = [cv_val_means[i] for i in sorted_idx]
            sorted_stds = [cv_val_stds[i] for i in sorted_idx]
            sorted_colors = [colors[i] for i in sorted_idx]

            bars = ax.barh(sorted_names, sorted_means, xerr=sorted_stds,
                           color=sorted_colors, edgecolor='white', capsize=4, height=0.6)
            ax.set_xlabel('CV Validation Accuracy (%)', fontsize=10)
            ax.set_title('Cross-Validation Accuracy ± Std Dev', fontsize=11, fontweight='bold')
            for i, (mean, std) in enumerate(zip(sorted_means, sorted_stds)):
                ax.text(mean + std + 0.3, i, f'{mean:.1f}', va='center', fontsize=9, fontweight='bold')
            ax.set_xlim(0, 110)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.markdown('<div class="section-title">Overfit Index (Train - Test Gap)</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            gaps = [t - v for t, v in zip(train_accs, test_accs)]
            gap_colors = ['#ef4444' if g > 5 else '#10b981' for g in gaps]
            sorted_gap_idx = np.argsort(gaps)[::-1]
            ax.barh([model_names[i] for i in sorted_gap_idx],
                    [gaps[i] for i in sorted_gap_idx],
                    color=[gap_colors[i] for i in sorted_gap_idx],
                    edgecolor='white', height=0.6)
            ax.axvline(x=5, color='#f59e0b', linestyle='--', linewidth=1.5, label='5% threshold')
            ax.set_xlabel('Train Acc − Test Acc (%)', fontsize=10)
            ax.set_title('Overfitting Gap per Model', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── Chart 3: Confusion matrix for selected model
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm_model = st.selectbox("Select model for confusion matrix:", model_names, key='cm_sel')
        y_pred = results[cm_model]['y_pred_test']

        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_te, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=range(10), yticklabels=range(10),
                    linewidths=0.5, linecolor='white')
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_title(f'{cm_model} — Confusion Matrix on Test Set', fontsize=12, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Chart 4: Per-class F1 heatmap across all models
        st.markdown('<div class="section-title">Per-Class F1 Score Heatmap</div>', unsafe_allow_html=True)
        f1_matrix = {}
        for m in model_names:
            rep = classification_report(results[m]['y_test'], results[m]['y_pred_test'],
                                        output_dict=True)
            f1_matrix[m] = {str(i): rep[str(i)]['f1-score'] for i in range(10)}

        f1_df = pd.DataFrame(f1_matrix).T
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(f1_df.astype(float)*100, annot=True, fmt='.1f', cmap='RdYlGn',
                    ax=ax, vmin=50, vmax=100, linewidths=0.5)
        ax.set_xlabel('Digit Class', fontsize=11)
        ax.set_ylabel('Model', fontsize=11)
        ax.set_title('Per-Class F1 Score (%) on Test Set', fontsize=12, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Chart 5: Learning curve for selected model (train size effect)
        st.markdown('<div class="section-title">Accuracy vs. Training Size (All Models)</div>', unsafe_allow_html=True)
        X_raw2, y_raw2, _ = load_data()

        @st.cache_data(show_spinner=False)
        def compute_learning_curves(use_pca_, pca_comp_):
            sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            curves = {m: {'train': [], 'test': []} for m in MODELS}
            X_full, y_full, _ = load_data()
            for sz in sizes:
                Xtr, Xte, ytr, yte = train_test_split(X_full, y_full, train_size=sz,
                                                       random_state=42, stratify=y_full)
                for nm, base_m in MODELS.items():
                    pipe = build_pipeline(base_m, use_pca_, pca_comp_)
                    pipe.fit(Xtr, ytr)
                    curves[nm]['train'].append(accuracy_score(ytr, pipe.predict(Xtr))*100)
                    curves[nm]['test'].append(accuracy_score(yte, pipe.predict(Xte))*100)
            return sizes, curves

        with st.spinner("Computing learning curves..."):
            sizes, curves = compute_learning_curves(use_pca, pca_components)

        fig, ax = plt.subplots(figsize=(12, 5))
        for nm in model_names:
            ax.plot([s*100 for s in sizes], curves[nm]['test'],
                    'o-', color=MODEL_COLORS[nm], linewidth=2, markersize=5, label=nm)
        ax.set_xlabel('Training Size (%)', fontsize=11)
        ax.set_ylabel('Test Accuracy (%)', fontsize=11)
        ax.set_title('Learning Curves — Test Accuracy vs Training Size', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

# ═══════════════════════════════════════════════
# TAB 4 — DRAW & CLASSIFY
# ═══════════════════════════════════════════════
with tab_draw:
    st.markdown('<div class="section-title">✏️ Draw a Digit and Classify It</div>', unsafe_allow_html=True)

    if 'results' not in st.session_state:
        st.warning("⚠️ Please train the models first (press **Train All Models** in the sidebar).")
    else:
        results = st.session_state['results']

        col_canvas, col_result = st.columns([1, 1])

        with col_canvas:
            st.markdown("**Draw a digit (0–9) in the canvas below:**")
            canvas_result = st_canvas(
                fill_color="rgba(255,255,255,0)",
                stroke_width=18,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=200,
                width=200,
                drawing_mode="freedraw",
                key="canvas",
            )

            classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)
            clear_note = st.caption("Clear the canvas by refreshing the page or clicking 'New Drawing'")

        with col_result:
            st.markdown("**Classification Results:**")

            if classify_btn and canvas_result.image_data is not None:
                img_data = canvas_result.image_data
                if img_data[:, :, :3].sum() > 100:  # not empty
                    try:
                        processed = preprocess_canvas_image(img_data)

                        # Show preprocessed 8x8
                        fig_prev, ax_prev = plt.subplots(figsize=(2.5, 2.5))
                        ax_prev.imshow(processed.reshape(8, 8), cmap='Blues', vmin=0, vmax=16)
                        ax_prev.set_title('Preprocessed (8×8)', fontsize=9)
                        ax_prev.axis('off')
                        fig_prev.tight_layout()
                        st.pyplot(fig_prev)
                        plt.close()

                        # Predict with all models
                        pred_df_rows = []
                        for name, res in results.items():
                            pipe = res['pipe']
                            pred = pipe.predict(processed)[0]
                            if hasattr(pipe, 'predict_proba'):
                                proba = pipe.predict_proba(processed)[0]
                                conf = proba.max() * 100
                            else:
                                conf = 100.0
                            pred_df_rows.append({'Model': name, 'Prediction': pred,
                                                  'Confidence': f'{conf:.1f}%'})

                        pred_df = pd.DataFrame(pred_df_rows)
                        st.dataframe(pred_df, use_container_width=True, hide_index=True)

                        # Majority vote
                        preds = [r['Prediction'] for r in pred_df_rows]
                        majority = pd.Series(preds).value_counts().idxmax()
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea, #764ba2);
                                    color: white; border-radius: 12px; padding: 1.2rem;
                                    text-align: center; margin-top: 1rem;'>
                            <div style='font-size: 3rem; font-weight: 900;'>{majority}</div>
                            <div style='font-size: 0.9rem; opacity: 0.85;'>Majority Vote Prediction</div>
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Processing error: {e}")
                else:
                    st.info("The canvas appears empty. Draw a digit first!")
            else:
                st.markdown("""
                <div style='text-align:center; padding: 3rem 1rem; color: #94a3b8;'>
                    <div style='font-size: 3rem;'>👆</div>
                    <div>Draw a digit on the left, then click <b>Classify</b></div>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#94a3b8; font-size:0.8rem;'>"
    "MNIST Classifier Studio · Built with Streamlit & scikit-learn"
    "</div>",
    unsafe_allow_html=True
)
