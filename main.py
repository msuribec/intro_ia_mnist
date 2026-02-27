import io
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

from sklearn.datasets import fetch_openml
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedShuffleSplit,
    cross_validate,
    learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from streamlit_drawable_canvas import st_canvas


# ----------------------------
# Data loading + quality checks
# ----------------------------
@st.cache_data(show_spinner=False)
def load_mnist_openml(sample_limit: int = 70000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads MNIST from OpenML via sklearn. Returns X (float32) and y (int64).
    Note: This is the standard MNIST (70k samples, 784 features).
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int64)

    if sample_limit is not None and sample_limit < X.shape[0]:
        # deterministic subset (no leakage risk; just slicing)
        X = X[:sample_limit]
        y = y[:sample_limit]

    return X, y


def data_quality_report(X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """
    Verifies basic data quality. Pure inspection only (no fitting -> no leakage).
    """
    report = {}
    report["X_shape"] = X.shape
    report["y_shape"] = y.shape
    report["X_dtype"] = str(X.dtype)
    report["y_dtype"] = str(y.dtype)

    # Missing / invalid checks
    report["missing_in_X"] = int(np.isnan(X).sum())
    report["missing_in_y"] = int(np.isnan(y).sum()) if np.issubdtype(y.dtype, np.floating) else 0
    report["inf_in_X"] = int(np.isinf(X).sum())
    report["min_pixel"] = float(np.min(X))
    report["max_pixel"] = float(np.max(X))

    # Label distribution
    labels, counts = np.unique(y, return_counts=True)
    report["class_distribution"] = dict(zip(labels.tolist(), counts.tolist()))
    report["num_classes"] = int(len(labels))
    report["unique_labels"] = labels.tolist()

    return report


# ----------------------------
# CV strategies (NO leakage)
# ----------------------------
def make_cv(strategy: str, n_splits: int, random_state: int):
    if strategy == "StratifiedKFold":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if strategy == "RepeatedStratifiedKFold":
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=2, random_state=random_state)
    if strategy == "ShuffleSplit":
        return ShuffleSplit(n_splits=max(10, n_splits), test_size=0.2, random_state=random_state)
    if strategy == "StratifiedShuffleSplit":
        return StratifiedShuffleSplit(n_splits=max(10, n_splits), test_size=0.2, random_state=random_state)
    raise ValueError(f"Unknown CV strategy: {strategy}")


# ----------------------------
# Models
# ----------------------------
@dataclass
class ModelSpec:
    name: str
    estimator: object


def get_model_specs(random_state: int) -> List[ModelSpec]:
    """
    You asked for: knn, bayes, svm, rf, dt, mc, ann.
    We'll train all 7 (so you can compare).
    """
    return [
        ModelSpec("KNN", KNeighborsClassifier(n_neighbors=5)),
        ModelSpec("Bayes (GaussianNB)", GaussianNB()),
        ModelSpec("SVM (RBF)", SVC(kernel="rbf", gamma="scale", C=3.0)),
        ModelSpec("Random Forest", RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)),
        ModelSpec("Decision Tree", DecisionTreeClassifier(random_state=random_state)),
        ModelSpec("Bayes (MultinomialNB)", MultinomialNB(alpha=1.0)),
        ModelSpec("ANN (MLP)", MLPClassifier(hidden_layer_sizes=(128,), max_iter=15, random_state=random_state)),
    ]


def build_pipeline(estimator, use_pca: bool, pca_components: int, for_multinomial_nb: bool) -> Pipeline:
    """
    IMPORTANT (no leakage):
    - ALL transforms (scaler / PCA) live inside the pipeline.
    - During CV, each fold fits transforms only on that fold’s training split.
    """
    steps = []

    # MultinomialNB expects non-negative features. MNIST pixels are 0..255 already.
    # Scaling to mean=0 would create negatives -> keep raw if multinomial.
    if not for_multinomial_nb:
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))

    if use_pca:
        steps.append(("pca", PCA(n_components=pca_components, random_state=0)))

    steps.append(("model", estimator))
    return Pipeline(steps)


# ----------------------------
# Training + evaluation
# ----------------------------
@st.cache_resource(show_spinner=False)
def train_and_evaluate(
    train_fraction: float,
    use_pca: bool,
    pca_components: int,
    cv_strategy: str,
    cv_splits: int,
    random_state: int,
    sample_limit: int,
) -> Dict[str, object]:
    """
    Returns trained pipelines and metrics.
    NO leakage:
    - We first split into (train, test).
    - We do cross-validation ONLY on train.
    - Then fit each pipeline on full train and evaluate on held-out test.
    """
    X, y = load_mnist_openml(sample_limit=sample_limit)

    # Hold-out test split (never touched by CV fitting)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # User-controlled training fraction (subsample ONLY from training split)
    if not (0.05 <= train_fraction <= 1.0):
        raise ValueError("train_fraction must be between 0.05 and 1.0")

    n_train = int(len(X_train_full) * train_fraction)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_train_full), size=n_train, replace=False)
    X_train = X_train_full[idx]
    y_train = y_train_full[idx]

    cv = make_cv(cv_strategy, n_splits=cv_splits, random_state=random_state)

    model_specs = get_model_specs(random_state=random_state)

    rows = []
    trained_pipelines = {}

    for spec in model_specs:
        is_multinomial = spec.name.startswith("Bayes (MultinomialNB)")
        pipe = build_pipeline(
            estimator=spec.estimator,
            use_pca=use_pca,
            pca_components=pca_components,
            for_multinomial_nb=is_multinomial,
        )

        # Cross-validate on TRAIN ONLY
        scoring = {"acc": "accuracy", "f1": "f1_macro"}
        cv_out = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )

        cv_train_acc = float(np.mean(cv_out["train_acc"]))
        cv_val_acc = float(np.mean(cv_out["test_acc"]))
        cv_train_f1 = float(np.mean(cv_out["train_f1"]))
        cv_val_f1 = float(np.mean(cv_out["test_f1"]))

        # Fit on full (subsampled) train, then evaluate on held-out test
        pipe.fit(X_train, y_train)
        y_pred_test = pipe.predict(X_test)

        test_acc = float(accuracy_score(y_test, y_pred_test))
        test_f1 = float(f1_score(y_test, y_pred_test, average="macro"))

        trained_pipelines[spec.name] = pipe

        rows.append(
            {
                "Model": spec.name,
                "CV Train Acc (mean)": cv_train_acc,
                "CV Val Acc (mean)": cv_val_acc,
                "CV Train F1-macro (mean)": cv_train_f1,
                "CV Val F1-macro (mean)": cv_val_f1,
                "Test Acc (holdout)": test_acc,
                "Test F1-macro (holdout)": test_f1,
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values(by="CV Val Acc (mean)", ascending=False).reset_index(drop=True)

    # Choose best by CV Val Acc
    best_model_name = str(metrics_df.loc[0, "Model"])
    best_pipe = trained_pipelines[best_model_name]

    # Confusion matrix for best model (on holdout test)
    y_pred_best = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best, labels=np.arange(10))

    # Learning curve for best model (on TRAIN ONLY)
    # We evaluate train/val scores as training size increases.
    train_sizes, train_scores, val_scores = learning_curve(
        best_pipe,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 6),
    )

    return {
        "X_shape": X.shape,
        "train_size_used": len(X_train),
        "test_size": len(X_test),
        "metrics_df": metrics_df,
        "trained_pipelines": trained_pipelines,
        "best_model_name": best_model_name,
        "confusion_matrix": cm,
        "learning_curve": {
            "train_sizes": train_sizes,
            "train_scores_mean": np.mean(train_scores, axis=1),
            "val_scores_mean": np.mean(val_scores, axis=1),
        },
        # Keep a tiny sample of test set for potential debug/visuals (not needed for training)
        "test_pack": (X_test[:2000], y_test[:2000]),
    }


# ----------------------------
# Drawing -> MNIST preprocessing
# ----------------------------
def canvas_to_mnist_vector(canvas_rgba: np.ndarray) -> np.ndarray:
    """
    Convert canvas RGBA (H,W,4) to MNIST-like 28x28 flattened vector in [0,255].
    Assumes user draws black on white background (default).
    """
    img = Image.fromarray(canvas_rgba.astype(np.uint8), mode="RGBA")
    img = img.convert("L")  # grayscale

    # Invert so that drawn strokes become "bright" like MNIST digits (white on black)
    img = ImageOps.invert(img)

    # Crop around content (optional but helpful)
    bbox = img.getbbox()
    if bbox is not None:
        img = img.crop(bbox)

    # Resize to 28x28
    img = img.resize((28, 28), resample=Image.Resampling.LANCZOS)

    # Convert to numpy
    arr = np.array(img).astype(np.float32)

    # Normalize to 0..255 (already should be)
    arr = np.clip(arr, 0, 255)

    return arr.reshape(1, -1)


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="MNIST Model Lab (No Leakage)", layout="wide")
st.title("MNIST Classification Lab (Streamlit)")

with st.expander("What this app guarantees (important)"):
    st.markdown(
        """
- **No data leakage**: the test set is split **first** and never used for cross-validation fitting.
- All preprocessing (scaling / PCA) is done **inside sklearn Pipelines**, so folds don’t “see” each other.
- The “% of training data” slider only subsamples from the training split (never from test).
"""
    )

# Sidebar controls
st.sidebar.header("Controls")

train_pct = st.sidebar.slider("Training data fraction (of training split)", min_value=5, max_value=100, value=30, step=5)
train_fraction = train_pct / 100.0

use_pca = st.sidebar.checkbox("Use PCA", value=True)
pca_components = st.sidebar.slider("PCA components", min_value=10, max_value=200, value=60, step=10, disabled=not use_pca)

cv_strategy = st.sidebar.selectbox(
    "Cross-validation strategy",
    ["StratifiedKFold", "RepeatedStratifiedKFold", "ShuffleSplit", "StratifiedShuffleSplit"],
    index=0,
)
cv_splits = st.sidebar.slider("CV splits (or base splits)", min_value=3, max_value=10, value=5, step=1)

sample_limit = st.sidebar.selectbox("Dataset size (speed control)", [70000, 30000, 15000, 8000], index=2)

random_state = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

run = st.sidebar.button("Run training", type="primary")

# Load + quality check
X, y = load_mnist_openml(sample_limit=sample_limit)
dq = data_quality_report(X, y)

col_a, col_b = st.columns([1, 1])
with col_a:
    st.subheader("Data quality checks")
    st.json(
        {
            "X_shape": dq["X_shape"],
            "y_shape": dq["y_shape"],
            "X_dtype": dq["X_dtype"],
            "y_dtype": dq["y_dtype"],
            "missing_in_X": dq["missing_in_X"],
            "inf_in_X": dq["inf_in_X"],
            "pixel_range": [dq["min_pixel"], dq["max_pixel"]],
            "num_classes": dq["num_classes"],
        }
    )

with col_b:
    st.subheader("Class distribution")
    dist = pd.DataFrame({"digit": list(dq["class_distribution"].keys()), "count": list(dq["class_distribution"].values())})
    st.dataframe(dist, use_container_width=True)

# Show a few sample images
with st.expander("Preview some MNIST samples"):
    n_show = 12
    idxs = np.random.default_rng(random_state).choice(len(X), size=n_show, replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    axes = axes.ravel()
    for ax, i in zip(axes, idxs):
        ax.imshow(X[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"y={y[i]}")
        ax.axis("off")
    st.pyplot(fig)

# Run training/eval
if run:
    with st.spinner("Training models + cross-validation (train-only) + holdout test evaluation..."):
        t0 = time.time()
        results = train_and_evaluate(
            train_fraction=train_fraction,
            use_pca=use_pca,
            pca_components=int(pca_components),
            cv_strategy=cv_strategy,
            cv_splits=int(cv_splits),
            random_state=int(random_state),
            sample_limit=int(sample_limit),
        )
        st.session_state["results"] = results
        st.session_state["trained_pipelines"] = results["trained_pipelines"]
        st.session_state["best_model_name"] = results["best_model_name"]
        st.session_state["use_pca"] = use_pca
        st.session_state["pca_components"] = int(pca_components)
        st.session_state["random_state"] = int(random_state)
        st.session_state["train_fraction"] = train_fraction
        st.session_state["cv_strategy"] = cv_strategy
        st.session_state["cv_splits"] = int(cv_splits)
        st.session_state["sample_limit"] = int(sample_limit)
        t1 = time.time()

    st.success(f"Done in {t1 - t0:.1f}s. Best model by CV Val Acc: {results['best_model_name']}")

# Display results if available
if "results" in st.session_state:
    results = st.session_state["results"]
    metrics_df = results["metrics_df"]

    st.subheader("Model performance summary")
    st.caption("CV metrics are computed **only on the training split**. Test metrics are on a **held-out** test set.")
    st.dataframe(metrics_df, use_container_width=True)

    # Bar chart: CV Val Acc vs Test Acc
    st.subheader("Graphs: CV vs Test performance")

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    x = np.arange(len(metrics_df))
    ax1.bar(x - 0.2, metrics_df["CV Val Acc (mean)"].values, width=0.4, label="CV Val Acc (mean)")
    ax1.bar(x + 0.2, metrics_df["Test Acc (holdout)"].values, width=0.4, label="Test Acc (holdout)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df["Model"].values, rotation=35, ha="right")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(x - 0.2, metrics_df["CV Val F1-macro (mean)"].values, width=0.4, label="CV Val F1 (macro)")
    ax2.bar(x + 0.2, metrics_df["Test F1-macro (holdout)"].values, width=0.4, label="Test F1 (macro)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_df["Model"].values, rotation=35, ha="right")
    ax2.set_ylabel("F1-macro")
    ax2.legend()
    st.pyplot(fig2)

    # Confusion matrix for best model
    st.subheader(f"Best model confusion matrix (holdout test): {results['best_model_name']}")
    cm = results["confusion_matrix"]
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    st.pyplot(fig_cm)

    # Learning curve (best model)
    st.subheader("Learning curve (best model, train-only CV)")
    lc = results["learning_curve"]
    fig_lc, ax_lc = plt.subplots(figsize=(7, 4))
    ax_lc.plot(lc["train_sizes"], lc["train_scores_mean"], marker="o", label="Train (CV mean)")
    ax_lc.plot(lc["train_sizes"], lc["val_scores_mean"], marker="o", label="Validation (CV mean)")
    ax_lc.set_xlabel("Training samples")
    ax_lc.set_ylabel("Accuracy")
    ax_lc.legend()
    st.pyplot(fig_lc)

# Drawing + prediction
st.divider()
st.header("Draw a digit and classify it")

if "trained_pipelines" not in st.session_state:
    st.info("Train the models first (click **Run training**) so the app can classify your drawing.")
else:
    model_names = list(st.session_state["trained_pipelines"].keys())
    default_best = st.session_state.get("best_model_name", model_names[0])
    chosen_model_name = st.selectbox("Choose a trained model for prediction", model_names, index=model_names.index(default_best))

    st.caption("Tip: draw a big digit in the center. The app will crop and resize to 28×28 like MNIST.")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=12,
        stroke_color="black",
        background_color="white",
        width=260,
        height=260,
        drawing_mode="freedraw",
        key="canvas",
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Classify drawing", type="primary"):
            if canvas_result.image_data is None:
                st.warning("Please draw a digit first.")
            else:
                vec = canvas_to_mnist_vector(canvas_result.image_data)

                pipe = st.session_state["trained_pipelines"][chosen_model_name]
                pred = int(pipe.predict(vec)[0])

                # Show probabilities if available
                proba = None
                if hasattr(pipe[-1], "predict_proba"):
                    try:
                        proba = pipe.predict_proba(vec)[0]
                    except Exception:
                        proba = None

                st.success(f"Prediction: **{pred}**")

                # Show processed 28x28 image
                img28 = vec.reshape(28, 28)
                figp, axp = plt.subplots(figsize=(3, 3))
                axp.imshow(img28, cmap="gray")
                axp.axis("off")
                axp.set_title("Processed 28×28")
                st.pyplot(figp)

                if proba is not None and len(proba) == 10:
                    prob_df = pd.DataFrame({"digit": np.arange(10), "probability": proba})
                    st.dataframe(prob_df.sort_values("probability", ascending=False), use_container_width=True)

    with col2:
        st.markdown(
            """
**No leakage reminder**  
Your drawing is classified using the already-trained pipeline (scaler/PCA/model).  
The test split was never used to fit preprocessing or model parameters.
"""
        )
