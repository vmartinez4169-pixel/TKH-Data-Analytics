def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot an annotated confusion matrix using Plotly.
    Returns: plotly Figure"""
    import plotly.graph_objects as go
    from sklearn.metrics import confusion_matrix
    from tkh_utils import PALETTE, base_layout

    cm = confusion_matrix(y_true, y_pred)
    labels = ["Negative", "Positive"]
    cm_max = cm.max() if cm.max() > 0 else 1

    annotations = []
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            # Dark text on light (low-count) cells, white text on dark (high-count) cells
            text_color = "white" if (cm[i][j] / cm_max) >= 0.5 else "#1a1a2e"
            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=str(cm[i][j]),
                    font=dict(color=text_color, size=16),
                    showarrow=False,
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale=[
                [0, PALETTE["background"]],
                [1, PALETTE["primary"]],
            ],
            showscale=False,
        ),
        layout=base_layout(
            title=title,
            xaxis_title="Predicted",
            yaxis_title="Actual",
        ),
    )
    # sklearn's confusion_matrix() puts row 0 (Negative) first — reverse the y-axis
    # so it renders on top, matching sklearn's row-0-on-top convention.
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(annotations=annotations)
    return fig


def plot_roc_curve(y_true, y_prob, title="ROC Curve"):
    """Plot ROC curve with AUC score using Plotly.
    Returns: plotly Figure"""
    import plotly.graph_objects as go
    from sklearn.metrics import roc_curve, auc
    from tkh_utils import PALETTE, base_layout

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure(layout=base_layout(
        title=f"{title} (AUC = {roc_auc:.3f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    ))
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        line=dict(color=PALETTE["primary"], width=2.5),
        name=f"ROC (AUC = {roc_auc:.3f})",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color=PALETTE["muted"], width=1.5, dash="dash"),
        name="Random classifier",
    ))
    return fig


def plot_feature_importance(feature_names, importances,
                             title="Feature Importance", top_n=15):
    """Plot horizontal bar chart of feature importances.
    Returns: plotly Figure"""
    import numpy as np
    import plotly.graph_objects as go
    from tkh_utils import PALETTE, base_layout

    indices = np.argsort(importances)[-top_n:]

    fig = go.Figure(
        data=go.Bar(
            x=importances[indices],
            y=[feature_names[i] for i in indices],
            orientation="h",
            marker_color=PALETTE["primary"],
        ),
        layout=base_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Feature",
        ),
    )
    return fig


def plot_learning_curve(train_sizes, train_scores, val_scores,
                        title="Learning Curve"):
    """Plot training and validation learning curves.
    Returns: plotly Figure"""
    import numpy as np
    import plotly.graph_objects as go
    from tkh_utils import PALETTE, base_layout

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    fig = go.Figure(layout=base_layout(
        title=title,
        xaxis_title="Training Set Size",
        yaxis_title="Score",
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        mode="lines",
        line=dict(color=PALETTE["primary"], width=2.5),
        name="Training score",
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_mean,
        mode="lines",
        line=dict(color=PALETTE["secondary"], width=2.5),
        name="Validation score",
    ))
    return fig
