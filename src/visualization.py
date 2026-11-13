import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd

def plot_target_distribution(df: pd.DataFrame):
    """Plots clean target count distribution."""
    fig = px.histogram(
        df, x='booking_status', color='booking_status',
        title="Booking Status Distribution",
        color_discrete_map={'Not_Canceled': '#00CC96', 'Canceled': '#EF553B'},
        template='plotly_dark'
    )
    fig.update_layout(showlegend=False, xaxis_title="Booking Status", yaxis_title="Count")
    return fig

def plot_lead_time_distribution(df: pd.DataFrame):
    """Plots lead time distribution split by target."""
    fig = px.histogram(
        df, x='lead_time', color='booking_status', marginal='box',
        title="Lead Time Density by Booking Status", barmode='overlay',
        color_discrete_map={'Not_Canceled': '#00CC96', 'Canceled': '#EF553B'},
        template='plotly_dark'
    )
    return fig

def plot_confusion_matrix_plotly(cm_matrix, model_name: str):
    """Generates an interactive Plotly confusion matrix heatmap."""
    x_labels = ['Not Canceled (0)', 'Canceled (1)']
    y_labels = ['Not Canceled (0)', 'Canceled (1)']
    
    fig = ff.create_annotated_heatmap(
        z=cm_matrix, x=x_labels, y=y_labels, colorscale='Blues', showscale=True
    )
    fig.update_layout(
        title=f"Confusion Matrix: {model_name}",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        template='plotly_dark'
    )
    return fig