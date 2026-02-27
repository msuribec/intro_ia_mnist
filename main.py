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
    Converts a freehand canvas drawing into an 8x8 array that matches 
    the statistical properties of the sklearn digits dataset.
    """
    # 1. Extract the alpha channel (drawing intensity: 0 is background, 255 is drawing)
    alpha_channel = img_array[:, :, 3]
    
    # 2. Find the bounding box of the drawing to crop out empty space
    rows = np.any(alpha_channel > 0, axis=1)
    cols = np.any(alpha_channel > 0, axis=0)
    
    # If the canvas is completely empty, return None
    if not np.any(rows) or not np.any(cols):
        return None
        
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Crop the image to the exact bounding box of the drawing
    cropped = alpha_channel[rmin:rmax+1, cmin:cmax+1]
    
    # 3. Pad the cropped image to make it perfectly square (preserves aspect ratio)
    h, w = cropped.shape
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    squared = np.pad(cropped, ((pad_h, size - h - pad_h), (pad_w, size - w - pad_w)), mode='constant')
    
    # 4. Add a uniform margin. 
    # The sklearn digits dataset has empty borders. We add a 30% margin.
    margin = int(size * 0.3)
    padded_squared = np.pad(squared, margin, mode='constant')
    
    # 5. Downsample to 8x8 using high-quality LANCZOS resampling (anti-aliasing)
    img_pil = Image.fromarray(padded_squared)
    img_8x8 = img_pil.resize((8, 8), Image.Resampling.LANCZOS)
    
    # 6. Convert to numpy array and scale from 0-255 to 0-16 (sklearn's format)
    img_8x8_array = np.array(img_8x8)
    scaled_img = (img_8x8_array / 255.0) * 16.0
    
    # 7. Flatten to 1x64 for the machine learning model
    return scaled_img.reshape(1, -1)


# 1. Load Data (Cached)
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

# 3. Model Training Logic (Heavily Cached for Deployment)
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

    if cv_strategy == "Standard K-Fold":
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
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
st.write("Draw a number from 0 to 9. The image will be processed to match the training data.")
canvas_col, result_col = st.columns(2)

with canvas_col:
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=12,
        stroke_color="white",
        background_color="black",
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="canvas",
    )

with result_col:
    if canvas_result.image_data is not None:
        # Pass the raw canvas drawing through our new DIP pipeline
        processed_input = preprocess_canvas_image(canvas_result.image_data)
        
        if processed_input is not None:
            prediction = pipeline.predict(processed_input)
            
            st.subheader("Prediction")
            st.markdown(f"## **{prediction[0]}**")
            
            if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                probabilities = pipeline.predict_proba(processed_input)
                prob_df = pd.DataFrame(probabilities, columns=[str(i) for i in range(10)])
                st.bar_chart(prob_df.T)
            else:
                st.write("Probabilities not available for this model type.")
