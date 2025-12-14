import marimo

__generated_with = "0.18.4"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    import pandas as pd

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        confusion_matrix,
        ConfusionMatrixDisplay,
        roc_auc_score,
    )
    import shap


    import sys

    sys.path.append("../src")
    from gutatlas.models.xgboost import XGBBinClassTuner
    from gutatlas.models.lightgbm import LGBMBinClassTuner
    from gutatlas.models.logreg import LogRegBinClassTuner
    from gutatlas.models.metrics import (
        plot_shap_importance,
        plot_roc_curve,
        get_roc_auc,
        plot_confusion_matrix,
    )

    import xgboost as xgb
    import lightgbm as lgb
    import joblib
    from tensorflow import keras
    import matplotlib.pyplot as plt
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    return (
        LGBMBinClassTuner,
        LogRegBinClassTuner,
        XGBBinClassTuner,
        get_roc_auc,
        joblib,
        keras,
        lgb,
        pd,
        plot_confusion_matrix,
        plot_roc_curve,
        plot_shap_importance,
        plt,
        roc_auc_score,
        shap,
        train_test_split,
        xgb,
    )


@app.cell
def _(pd, train_test_split):
    gi_training = pd.read_parquet("../data/processed/microbiomap/gi_binclass_training_set.parquet")
    X = gi_training.drop(columns=["disease_present"])
    y = gi_training["disease_present"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LightGBM
    """)
    return


@app.cell
def _(LGBMBinClassTuner, X_train, y_train):
    lgbm_tuner = LGBMBinClassTuner(cv_splits=5, n_iter=10, n_jobs=-1)
    lgbm_tuner.fit(X_train, y_train)
    lgbm_tuner.best_params(), lgbm_tuner.best_score()
    return (lgbm_tuner,)


@app.cell
def _(lgbm_tuner):
    lgbm_tuner.save_model("../saved_models", "gi_bin_class_lightgbm_model.json")
    lgbm_tuner.save_params("../params", "gi_bin_class_lightgbm_params.json")
    return


@app.cell
def _(X_test, lgb, plot_confusion_matrix, y_test):
    lgbm_model = lgb.Booster(model_file="../saved_models/gi_bin_class_lightgbm_model.json")
    lgbm_preds_proba = lgbm_model.predict(X_test)
    lgbm_preds = (lgbm_preds_proba > 0.5).astype(int)  
    plot_confusion_matrix(y_test, lgbm_preds)
    return (lgbm_model,)


@app.cell
def _(X_test, get_roc_auc, lgbm_model, plot_roc_curve, y_test):
    lgbm_y_proba = lgbm_model.predict(X_test)
    lgbm_roc_auc = get_roc_auc(lgbm_model, y_test, lgbm_y_proba)
    print(lgbm_roc_auc)
    plot_roc_curve(y_test, lgbm_y_proba, lgbm_roc_auc).show()
    return


@app.cell
def _(X_test, lgbm_model, plot_shap_importance):
    plot_shap_importance(lgbm_model, X_test, max_display=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## logistic regression
    """)
    return


@app.cell
def _(LogRegBinClassTuner, X_train, y_train):
    logreg_tuner = LogRegBinClassTuner(cv_splits=5, n_iter=10, n_jobs=-1)
    logreg_tuner.fit(X_train, y_train)
    logreg_tuner.best_params(), logreg_tuner.best_score()
    return (logreg_tuner,)


@app.cell
def _(logreg_tuner):
    logreg_tuner.save_model("../saved_models", "gi_bin_class_logreg_model.pkl")
    logreg_tuner.save_params("../params", "gi_bin_class_logreg_params.json")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### confusion matrix
    """)
    return


@app.cell
def _(X_test, joblib, plot_confusion_matrix, y_test):
    logreg_model = joblib.load("../saved_models/gi_bin_class_logreg_model.pkl")
    logreg_preds = logreg_model.predict(X_test)
    plot_confusion_matrix(y_test, logreg_preds)
    return (logreg_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### roc auc
    """)
    return


@app.cell
def _(X_test, get_roc_auc, logreg_model, plot_roc_curve, y_test):
    logreg_y_proba = logreg_model.predict_proba(X_test)[:, 1]
    logreg_roc_auc = get_roc_auc(logreg_model, y_test, logreg_y_proba)
    print(logreg_roc_auc)
    plot_roc_curve(y_test, logreg_y_proba, logreg_roc_auc).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### feature coefficient interpretation
    """)
    return


@app.cell
def _(X_train, logreg_model, pd, plt):
    # Extract and interpret feature coefficients from logistic regression
    import numpy as np
    coefficients = logreg_model.coef_[0]
    feature_names = X_train.columns
    # Get coefficients
    coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients, 'abs_coefficient': np.abs(coefficients)}).sort_values('abs_coefficient', ascending=False)
    protective_features = coef_df[coef_df['coefficient'] < 0].head(20)
    risk_features = coef_df[coef_df['coefficient'] > 0].head(20)
    # Create DataFrame of features and their coefficients
    print('=' * 80)
    print('LOGISTIC REGRESSION FEATURE INTERPRETATION')
    print('=' * 80)
    print('\n' + '=' * 80)
    print('TOP 20 PROTECTIVE FEATURES (Negative Coefficients)')
    print('=' * 80)
    # Split into protective and risk-enhancing
    print('These genera are associated with REDUCED disease risk')
    print('-' * 80)
    for idx, row in protective_features.iterrows():
        print(f'{row['feature']:70s} | Coef: {row['coefficient']:8.4f}')
    print('\n' + '=' * 80)
    print('TOP 20 RISK-ENHANCING FEATURES (Positive Coefficients)')
    print('=' * 80)
    print('These genera are associated with INCREASED disease risk')
    print('-' * 80)
    for idx, row in risk_features.iterrows():
        print(f'{row['feature']:70s} | Coef: {row['coefficient']:8.4f}')
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    protective_features_plot = protective_features.head(15)
    axes[0].barh(range(len(protective_features_plot)), protective_features_plot['coefficient'], color='green', alpha=0.7)
    axes[0].set_yticks(range(len(protective_features_plot)))
    axes[0].set_yticklabels([feat.replace('bacteria_', '').replace('_', ' ') for feat in protective_features_plot['feature']], fontsize=8)
    axes[0].set_xlabel('Coefficient (Negative = Protective)', fontsize=10, fontweight='bold')
    axes[0].set_title('Top 15 Protective Features\n(Associated with Reduced Disease Risk)', fontsize=12, fontweight='bold')
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].invert_yaxis()
    risk_features_plot = risk_features.head(15)
    axes[1].barh(range(len(risk_features_plot)), risk_features_plot['coefficient'], color='red', alpha=0.7)
    # Visualize top features
    axes[1].set_yticks(range(len(risk_features_plot)))
    axes[1].set_yticklabels([feat.replace('bacteria_', '').replace('_', ' ') for feat in risk_features_plot['feature']], fontsize=8)
    # Protective features
    axes[1].set_xlabel('Coefficient (Positive = Risk-Enhancing)', fontsize=10, fontweight='bold')
    axes[1].set_title('Top 15 Risk-Enhancing Features\n(Associated with Increased Disease Risk)', fontsize=12, fontweight='bold')
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].invert_yaxis()
    plt.tight_layout()
    plt.show()
    print('\n' + '=' * 80)
    print('SUMMARY STATISTICS')
    print('=' * 80)
    # Risk-enhancing features
    print(f'Total features: {len(coefficients)}')
    print(f'Protective features (negative coef): {(coefficients < 0).sum()}')
    print(f'Risk-enhancing features (positive coef): {(coefficients > 0).sum()}')
    print(f'Mean absolute coefficient: {np.abs(coefficients).mean():.4f}')
    print(f'Max protective coefficient: {coefficients.min():.4f}')
    # Summary statistics
    print(f'Max risk-enhancing coefficient: {coefficients.max():.4f}')
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## deep learning approach

    ### Denoising Autoencoder (DAE) for feature extraction
    """)
    return


@app.cell
def _(X_test, X_train, keras, np):
    # Simple Denoising Autoencoder for unsupervised feature learning
    noise_factor = 0.2
    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    # Add noise to input
    X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
    input_dim = X_train.shape[1]
    encoding_dim = 64
    dae_encoder = keras.Sequential([keras.layers.Input(shape=(input_dim,)), keras.layers.Dense(256, activation='relu'), keras.layers.Dropout(0.2), keras.layers.Dense(128, activation='relu'), keras.layers.Dense(encoding_dim, activation='relu', name='encoding')])
    # Build autoencoder
    dae_decoder = keras.Sequential([keras.layers.Input(shape=(encoding_dim,)), keras.layers.Dense(128, activation='relu'), keras.layers.Dropout(0.2), keras.layers.Dense(256, activation='relu'), keras.layers.Dense(input_dim, activation='linear')])
    dae_autoencoder = keras.Sequential([dae_encoder, dae_decoder])  # Compressed representation
    dae_autoencoder.compile(optimizer='adam', loss='mse')
    # Encoder
    print('Training autoencoder...')
    dae_autoencoder.fit(X_train_noisy, X_train, epochs=50, batch_size=128, validation_split=0.2, verbose=1)
    X_train_encoded = dae_encoder.predict(X_train, verbose=0)
    X_test_encoded = dae_encoder.predict(X_test, verbose=0)
    print(f'\nOriginal features: {X_train.shape[1]}')
    print(f'Encoded features: {X_train_encoded.shape[1]}')
    # Decoder
    # Full autoencoder
    # Train (reconstruct original from noisy input)
    # Extract compressed features
    print(f'Compression ratio: {X_train.shape[1] / X_train_encoded.shape[1]:.1f}x')
    return X_test_encoded, X_train_encoded, dae_encoder, encoding_dim


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Supervised classifier on encoded features
    """)
    return


@app.cell
def _(
    X_test_encoded,
    X_train_encoded,
    encoding_dim,
    keras,
    plot_confusion_matrix,
    roc_auc_score,
    y_test,
    y_train,
):
    # Train classifier on compressed features
    dae_classifier = keras.Sequential([
        keras.layers.Input(shape=(encoding_dim,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    dae_classifier.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    print("Training classifier on encoded features...")
    dae_history = dae_classifier.fit(
        X_train_encoded, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max')],
        verbose=1
    )

    # Evaluate
    dae_preds_proba = dae_classifier.predict(X_test_encoded, verbose=0).flatten()
    dae_preds = (dae_preds_proba > 0.5).astype(int)
    dae_auc = roc_auc_score(y_test, dae_preds_proba)

    print(f"\nDAE + Classifier ROC AUC: {dae_auc:.4f}")
    plot_confusion_matrix(y_test, dae_preds)
    return (dae_classifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SHAP interpretation (encoder + classifier)
    """)
    return


@app.cell
def _(X_test, X_train, dae_classifier, dae_encoder, np, shap):
    # SHAP interpretation for DAE + classifier
    print("Computing SHAP values for DAE model...")
    print("This shows which original features are most important")

    # Create a prediction function that goes through encoder + classifier
    def dae_predict_fn(X):
        """Full pipeline prediction."""
        # Ensure X is float numpy array
        X = np.asarray(X, dtype=np.float32)
        X_encoded = dae_encoder.predict(X, verbose=0)
        predictions = dae_classifier.predict(X_encoded, verbose=0).flatten()
        return predictions

    # Use KernelExplainer (model-agnostic, works with any model)
    # Convert to float numpy array explicitly
    dae_background = np.asarray(X_train[:100], dtype=np.float32)

    print("Testing prediction function on background data...")
    dae_test_pred = dae_predict_fn(dae_background[:5])
    print(f"✓ Prediction function works! Sample predictions: {dae_test_pred[:3]}")

    print("\nInitializing SHAP explainer...")
    dae_explainer = shap.KernelExplainer(dae_predict_fn, dae_background)

    # Compute SHAP values (using subset for speed)
    print("Computing SHAP values (this may take a few minutes)...")
    dae_X_test_sample = np.asarray(X_test[:200], dtype=np.float32)
    dae_shap_values = dae_explainer.shap_values(dae_X_test_sample, nsamples=100)

    # Plot (use DataFrame for feature names if available)
    print("\nTop 20 most important original features:")
    if hasattr(X_test, 'columns'):
        shap.summary_plot(dae_shap_values, X_test[:200], max_display=20, show=True)
    else:
        shap.summary_plot(dae_shap_values, dae_X_test_sample, max_display=20, show=True)
    return


@app.cell
def _(X_train, keras, y_train):
    nn_model = keras.Sequential(
        [
            keras.layers.Input((X_train.shape[1],)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    nn_model.compile(
        loss="binary_crossentropy",
        metrics=["accuracy"],
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
    )

    nn_history = nn_model.fit(
        X_train, y_train, epochs=100, batch_size=1028, validation_split=0.1, verbose=1
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## XGBoost
    """)
    return


@app.cell
def _(XGBBinClassTuner, X_train, y_train):
    xgb_tuner = XGBBinClassTuner(cv_splits=5, n_iter=10, n_jobs=-1)
    xgb_tuner.fit(X_train, y_train)
    xgb_tuner.best_params(), xgb_tuner.best_score()
    return (xgb_tuner,)


@app.cell
def _(xgb_tuner):
    xgb_tuner.save_model("../saved_models", "gi_bin_class_xgboost_model.json")
    xgb_tuner.save_params("../params", "gi_bin_class_xgboost_params.json")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### confusion matrix
    """)
    return


@app.cell
def _(X_test, plot_confusion_matrix, xgb, y_test):
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("../saved_models/gi_bin_class_xgboost_model.json")
    xgb_preds = xgb_model.predict(X_test)
    plot_confusion_matrix(y_test, xgb_preds)
    return (xgb_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### shap
    """)
    return


@app.cell
def _(X_test, plot_shap_importance, xgb_model):
    plot_shap_importance(xgb_model, X_test, max_display=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### roc auc
    """)
    return


@app.cell
def _(X_test, get_roc_auc, plot_roc_curve, xgb_model, y_test):
    xgb_y_proba = xgb_model.predict_proba(X_test)
    xgb_roc_auc = get_roc_auc(xgb_model, y_test, xgb_y_proba[:, 1])
    print(xgb_roc_auc)
    plot_roc_curve(y_test, xgb_y_proba[:, 1], xgb_roc_auc).show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
