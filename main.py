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

# --- Fixed Digital Image Processing (DIP) Pipeline ---
def preprocess_canvas_image(img_array):
    """
    Converts canvas drawing to an 8x8 array matching sklearn digits.
    """
    alpha_channel = img_array[:, :, 3]
    
    rows = np.any(alpha_channel > 0, axis=1)
    cols = np.any(alpha_channel > 0, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None, None
        
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = alpha_channel[rmin:rmax+1, cmin:cmax+1]
    
    h, w = cropped.shape
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    squared = np.pad(cropped, ((pad_h, size - h - pad_h), (pad_w, size - w - pad_w)), mode='constant')
    
    # 25% margin replicates the 1-2 pixel border in sklearn's 8x8 digits
    margin = int(size * 0.25)
    padded = np.pad(squared, margin, mode='constant')
    
    # Use BILINEAR instead of LANCZOS to prevent ringing artifacts on small grids
    img_pil = Image.fromarray(padded)
    img_8x8 = img_pil.resize((8, 8), Image.Resampling.BILINEAR)
    img_8x8_array = np.array(img_8x8, dtype=float)
    
    # Normalize intensity: Force the brightest stroke to equal exactly 16.0
    if img_8x8_array.max() > 0:
        img_8x8_array = (img_8x8_array / img_8x8_array.max()) * 16.0
        
    return img_8x8_array.reshape(1, -1), img_8x8_array


# 1. Load Data
@st.cache_data
def get_data():
    digits = load_digits()
    return digits.data, digits.target

X, y = get_data()

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

# 4. Interactive Canvas
st.header("Draw a Digit")
st.write("Draw a large, centered number from 0 to 9.")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.write("1. Draw Here:")
    # Increased stroke width helps the drawing survive the downsample
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.write("2. Processed 8x8 Input:")
    if canvas_result.image_data is not None:
        processed_input, img_8x8_viz = preprocess_canvas_image(canvas_result.image_data)
        
        if img_8x8_viz is not None:
            # Visualize what the model actually sees
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(img_8x8_viz, cmap='gray', vmin=0, vmax=16)
            ax.axis('off')
            st.pyplot(fig)

with col3:
    st.write("3. Model
