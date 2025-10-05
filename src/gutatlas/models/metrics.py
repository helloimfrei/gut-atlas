import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay
import shap

def plot_pred_scatter(y_true, y_pred,opacity=0.3):
    """Plot predicted values vs true values."""
    pred_scatter = px.scatter(
        x=y_true,
        y=y_pred,
        labels={'x': 'True Values', 'y': 'Predicted Values'},
    )

    identity = go.Scatter(
        x=np.linspace(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()), 100),
        y=np.linspace(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()), 100),
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="y = x",
    )
    pred_scatter.add_trace(identity)
    pred_scatter.update_traces(marker=dict(size=5, opacity=opacity))
    return pred_scatter


def regression_metrics(y_test, y_pred):
    """Calculate mean squared error, root mean squared error, and mean absolute error."""
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    return {'mse': mse, 'rmse': rmse, 'mae': mae}


def plot_confusion_matrix(y_test, y_pred, display_labels=['no_disease','disease']):
    """Plot confusion matrix for given true labels and predicted labels."""
    cm = confusion_matrix(y_test, y_pred)
    cm_disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    cm_disp.plot()
    return plt


def plot_shap_importance(model, X_test,max_display=20, feature_names=None):
    """Plot SHAP feature importance for given model and test data."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test) 
    return shap.summary_plot(shap_values, X_test,feature_names=feature_names, max_display=max_display)

def get_roc_auc(model, y_test, y_proba):
    roc_auc = roc_auc_score(y_test,y_proba)
    return roc_auc

def plot_roc_curve(y_test, y_proba, roc_auc):
    """Plot ROC curve for given true labels and predicted probabilities."""
    fpr, tpr, thresholds = roc_curve(y_test,y_proba)
    roc_disp = RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc=roc_auc)
    roc_disp.plot()
    return plt
