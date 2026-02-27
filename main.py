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

# Models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Classifier App", layout="wide")
st.title("MNIST Digit Classification Toolkit")

# 1. Load Data
@st.cache_data
def get_data():
    # Using load_digits (8x8 version of MNIST) for faster Streamlit execution
    digits = load_digits()
    X = digits.data
    y = digits.target
    return X, y

X, y = get_data()

# 2. Data Quality Verification
st.header("1. Data Quality & Preprocessing")
st.write(f"Dataset Shape: `{X.shape[0]}` samples, `{X.shape[1]}` features (8x8 pixels flattened).")

# Verification: Check for NaNs
missing_values = np.isnan(X).sum()
st.write(f"Missing Values: `{missing_values}`")
if missing_values == 0:
    st.success("Data Quality Check Passed: No missing values detected. Data is ready for splitting.")

# 3. User Controls (Sidebar)
st.sidebar.header("Configuration")
train_size_pct = st.sidebar.slider("Training Data Percentage", min_value=50, max_value=90, value=80, step=5)
use_pca = st.sidebar.checkbox("Use PCA for Dimensionality Reduction", value=False)
cv_strategy_name = st.sidebar.selectbox("Cross-Validation Strategy", ["Stratified K-Fold", "Standard K-Fold"])
model_name = st.sidebar.selectbox("Select Classification Model", 
    ["Support Vector Machine (SVM)", "Random Forest (RF)", "K-Nearest Neighbors (KNN)", 
     "Naive Bayes", "Artificial Neural Network (ANN)", "Decision Tree (DT)"]
)

# 4. Splitting Data
# Done BEFORE preprocessing to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    train_size=train_size_pct/100.0, 
    random_state=42, 
    stratify=y  # Ensure balanced class distribution in splits
)

# 5. Pipeline Construction (Zero Data Leakage Guarantee)
steps = [('scaler', StandardScaler())]

if use_pca:
    # Retain 95% of variance
    steps.append(('pca', PCA(n_components=0.95, random_state=42)))

# Select Model
models = {
    "Support Vector Machine (SVM)": SVC(probability=True, random_state=42),
    "Random Forest (RF)": RandomForestClassifier(random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Artificial Neural Network (ANN)": MLPClassifier(max_iter=1000, random_state=42),
    "Decision Tree (DT)": DecisionTreeClassifier(random_state=42)
}

selected_model = models[model_name]
steps.append(('classifier', selected_model))

# Create the pipeline
pipeline = Pipeline(steps)

# 6. Cross Validation
st.header("2. Model Training & Cross-Validation")

if cv_strategy_name == "Standard K-Fold":
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
else:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

with st.spinner("Running Cross-Validation..."):
    # CV is applied to the pipeline on the training data ONLY
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')

# Fit pipeline on the entire training set for final evaluation
pipeline.fit(X_train, y_train)

# Predictions
train_preds = pipeline.predict(X_train)
test_preds = pipeline.predict(X_test)

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

# 7. Visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cross-Validation Scores (5 Folds)")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"Fold {i+1}" for i in range(len(cv_scores))], cv_scores, color='skyblue')
    ax.axhline(cv_scores.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {cv_scores.mean():.3f}')
    ax.set_ylim([0, 1.1])
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Train vs Test Performance")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(["Training Set", "Test Set"], [train_acc, test_acc], color=['#4CAF50', '#FF9800'])
    ax2.set_ylim([0, 1.1])
    ax2.set_ylabel("Accuracy")
    for i, v in enumerate([train_acc, test_acc]):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha='center')
    st.pyplot(fig2)

# 8. Interactive Prediction (Drawing Canvas)
st.header("3. Draw a Digit")
st.write("Draw a number from 0 to 9 in the box below. The model will try to classify it.")

canvas_col, result_col = st.columns(2)

with canvas_col:
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="canvas",
    )

with result_col:
    if canvas_result.image_data is not None:
        # Get RGBA array from canvas
        img_array = canvas_result.image_data
        
        # Convert to grayscale PIL image
        img_pil = Image.fromarray(img_array.astype('uint8'), 'RGBA').convert('L')
        
        # Resize to 8x8 (the size used by sklearn's load_digits)
        img_resized = img_pil.resize((8, 8), Image.Resampling.LANCZOS)
        
        # Convert back to numpy array
        img_8x8 = np.array(img_resized)
        
        # Sklearn digits scale is 0 to 16
        img_scaled = (img_8x8 / 255.0) * 16.0
        
        # Flatten the 8x8 array to 1x64 for the model
        user_input_flattened = img_scaled.reshape(1, -1)
        
        # Predict using the fitted pipeline
        prediction = pipeline.predict(user_input_flattened)
        probabilities = pipeline.predict_proba(user_input_flattened)
        
        st.subheader("Prediction")
        st.markdown(f"## **{prediction[0]}**")
        
        st.write("Prediction Probabilities:")
        prob_df = pd.DataFrame(probabilities, columns=[str(i) for i in range(10)])
        st.bar_chart(prob_df.T)
