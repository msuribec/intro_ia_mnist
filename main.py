import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Classifier App", layout="wide")
st.title("MNIST Digit Classification Toolkit")

# --- Digital Image Processing (DIP) Pipeline ---
def preprocess_canvas_image(img_array):
    """
    Converts canvas drawing to an 8x8 array matching sklearn digits.
    """
    # 1. Extract the Red channel (0: background, 255: drawing)
    drawing = img_array[:, :, 0]
    
    if not np.any(drawing > 0):
        return None, None
        
    # 2. Bounding box crop
    rows = np.any(drawing > 0, axis=1)
    cols = np.any(drawing > 0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = drawing[rmin:rmax+1, cmin:cmax+1]
    
    # 3. Pad to perfectly square (preserves aspect ratio)
    h, w = cropped.shape
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    squared = np.pad(cropped, ((pad_h, size - h - pad_h), (pad_w, size - w - pad_w)), mode='constant')
    
    # 4. Small margin (15% creates a ~1 pixel border in 8x8 space)
    margin = int(size * 0.15)
    padded = np.pad(squared, margin, mode='constant')
    
    # 5. Downsample using BOX (averages the area, preserving thin strokes)
    img_pil = Image.fromarray(padded.astype(np.uint8))
    img_8x8 = img_pil.resize((8, 8), Image.Resampling.BOX)
    img_8x8_array = np.array(img_8x8, dtype=float)
    
    # 6. Normalize exactly to 0-16. DO NOT THRESHOLD.
    if img_8x8_array.max() > 0:
        img_8x8_array = (img_8x8_array / img_8x8_array.max()) * 16.0
        
    return img_8x8_array.reshape(1, -1), img_8x8_array

# 1. Load Data
@st.cache_data
def get_data():
    digits = load_digits()
    return digits.data, digits.target

X, y = get_data()

st.header("1. Data Quality & Preprocessing")
st.write(f"Dataset Shape: `{X.shape[0]}` samples, `{X.shape[1]}` features (8x8 pixels flattened).")

missing_values = np.isnan(X).sum()
if missing_values == 0:
    st.success("Data Quality Check Passed: No missing values detected.")

# 2. User Controls 
st.sidebar.header("Configuration")
train_size_pct = st.sidebar.slider("Training Data Percentage", min_value=50, max_value=90, value=80, step=5)
use_pca = st.sidebar.checkbox("Use PCA for Dimensionality Reduction", value=False)
cv_strategy_name = st.sidebar.selectbox("Cross-Validation Strategy", ["Stratified K-Fold", "Standard K-Fold"])
model_name = st.sidebar.selectbox("Select Classification Model", 
    ["Support Vector Machine (SVM)", "Random Forest (RF)", "K-Nearest Neighbors (KNN)", 
     "Naive Bayes", "Artificial Neural Network (ANN)", "Decision Tree (DT)"]
)

# 3. Model Training Logic
@st.cache_resource(show_spinner="Training model and running CV...")
def train_and_evaluate(train_size, use_pca, cv_strategy, model_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size/100.0, random_state=42, stratify=y
    )

    steps = [('scaler', StandardScaler())]
    if use_pca:
        steps.append(('pca', PCA(n_components=0.95, random_state=42)))

    models = {
        "Support Vector Machine (SVM)": SVC(probability=True, random_state=42),
        "Random Forest (RF)": RandomForestClassifier(random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Artificial Neural Network (ANN)": MLPClassifier(max_iter=1000, random_state=42),
        "Decision Tree (DT)": DecisionTreeClassifier(random_state=42)
    }
    
    pipeline = Pipeline(steps + [('classifier', models[model_name])])

    cv = KFold(n_splits=5, shuffle=True, random_state=42) if cv_strategy == "Standard K-Fold" else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')

    pipeline.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    test_acc = accuracy_score(y_test, pipeline.predict(X_test))

    return pipeline, cv_scores, train_acc, test_acc

pipeline, cv_scores, train_acc, test_acc = train_and_evaluate(
    train_size_pct, use_pca, cv_strategy_name, model_name
)

# 4. Visualizations
st.header("2. Model Performance")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cross-Validation (5 Folds)")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"Fold {i+1}" for i in range(len(cv_scores))], cv_scores, color='skyblue')
    ax.axhline(cv_scores.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {cv_scores.mean():.3f}')
    ax.set_ylim([0, 1.1])
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Train vs Test Accuracy")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    bars = ax2.bar(["Training Set", "Test Set"], [train_acc, test_acc], color=['#4CAF50', '#FF9800'])
    ax2.set_ylim([0, 1.1])
    ax2.set_ylabel("Accuracy")
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center')
    st.pyplot(fig2)

# 5. Interactive Canvas
st.header("3. Draw a Digit")
st.write("Draw a large, centered number from 0 to 9.")

col_canvas, col_viz, col_pred = st.columns([1, 1, 1])

with col_canvas:
    st.write("**1. Draw Here:**")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=25,
        stroke_color="white",
        background_color="black",
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="canvas",
    )

with col_viz:
    st.write("**2. Processed 8x8 Input:**")
    if canvas_result.image_data is not None:
        processed_input, img_8x8_viz = preprocess_canvas_image(canvas_result.image_data)
        
        if img_8x8_viz is not None:
            fig3, ax3 = plt.subplots(figsize=(2, 2))
            ax3.imshow(img_8x8_viz, cmap='gray', vmin=0, vmax=16)
            ax3.axis('off')
            fig3.patch.set_facecolor('black')
            st.pyplot(fig3)

with col_pred:
    st.write("**3. Model Output:**")
    if canvas_result.image_data is not None and 'processed_input' in locals() and processed_input is not None:
        prediction = pipeline.predict(processed_input)
        
        st.markdown(f"<h1 style='text-align: center; color: #4CAF50; font-size: 5rem;'>{prediction[0]}</h1>", unsafe_allow_html=True)
        
        if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
            probabilities = pipeline.predict_proba(processed_input)
            prob_df = pd.DataFrame(probabilities, columns=[str(i) for i in range(10)])
            st.bar_chart(prob_df.T, height=150)
        else:
            st.write("Probabilities not available for this model type.")
