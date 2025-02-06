import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import plotly.express as px


def plot_characterization(session_path, bins=64):
    figures, feat_available = {}, {}
    histogram_data = {}
    csv_data = {}
    module_path = os.path.join(session_path, 'Characterization')

    if os.path.exists(module_path):
        # Load histogram and CSV files
        for dataset in os.listdir(module_path):
            dataset_path = os.path.join(module_path , dataset)
            if os.path.isdir(dataset_path):
                hist_path = os.path.join(dataset_path, "histogram.npy")
                csv_path = os.path.join(dataset_path, "data_characterization.csv")
                rad_path = os.path.join(dataset_path, "rad_feat.csv")
                if os.path.exists(rad_path):
                    feature_names = pd.read_csv(rad_path).columns.tolist()[1:]  # Exclude the first column (e.g., 'Dataset')
                    feat_available[dataset] = {'rad_path': rad_path, 'feat_names':feature_names}

                if os.path.exists(hist_path) and os.path.exists(csv_path):
                    histogram_data[dataset] = np.load(hist_path)
                    csv_data[dataset] = pd.read_csv(csv_path)
                else:
                    return figures, feat_available
            else:
                return figures, feat_available
    else:
        return figures, feat_available

    colors = ["blue", "pink", "green", "purple", "orange", "red", "cyan", "magenta"]

    # Histogram Plot
    fig_hist = go.Figure()
    for idx, (dataset, hist_data) in enumerate(histogram_data.items()):
        bin_edges = np.linspace(-1024, 3071, num=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        color = colors[idx % len(colors)]

        fig_hist.add_trace(go.Bar(
            x=bin_centers, y=hist_data,
            name=dataset,
            marker=dict(color=color),
            opacity=0.5
        ))

    fig_hist.update_layout(
        title="Histograms",
        xaxis_title="Value",
        yaxis_title="Frequency",
        xaxis=dict(range=[-1024, 3071]),
        yaxis=dict(type='log'),
        barmode="overlay",
        template="plotly_white",
        legend_title="Datasets",
    )
    figures['histogram'] = fig_hist.to_html()

    # KDE Plot
    fig_kde = go.Figure()
    for idx, (dataset, hist_data) in enumerate(histogram_data.items()):
        kde = gaussian_kde(hist_data)
        x_range = np.linspace(np.min(hist_data), np.max(hist_data), 500)
        color = colors[idx % len(colors)]

        fig_kde.add_trace(go.Scatter(
            x=x_range, y=kde(x_range),
            mode="lines",
            name=dataset,
            line=dict(color=color)
        ))

    fig_kde.update_layout(
        title="Kernel Density Estimation (KDE)",
        xaxis_title="Intensity Value",
        yaxis_title="Density",
        template="plotly_white"
    )
    figures['kde'] = fig_kde.to_html()

    # Violin Plots for Skewness and Kurtosis
    for metric in ['skewness', 'kurtosis']:
        fig = go.Figure()
        for idx, (dataset, df) in enumerate(csv_data.items()):
            if metric in df.columns:
                fig.add_trace(go.Violin(y=df[metric], name=dataset, box_visible=True, line_color=colors[idx % len(colors)]))

        fig.update_layout(
            title=f"{metric.title()}",
            yaxis_title=metric.title(),
            template="plotly_white",
        )
        figures[metric] = fig.to_html()

    # Bar Plots for Categorical Features
    categorical_features = ['slice_thickness', 'convolution_kernel', 'manufacturer']
    for feature in categorical_features:
        fig = go.Figure()
        all_values = set()
        for dataset, df in csv_data.items():
            if feature in df.columns:
                all_values.update(df[feature].dropna().unique())

        all_values = sorted(all_values)
        for idx, (dataset, df) in enumerate(csv_data.items()):
            if feature in df.columns:
                counts = df[feature].value_counts().reindex(all_values, fill_value=0)
                fig.add_trace(go.Bar(
                    x=all_values, y=counts,
                    name=dataset,
                    marker_color=colors[idx % len(colors)]
                ))

        fig.update_layout(
            title=f"Distribution of {feature.replace('_', ' ').title()}",
            xaxis_title=feature.replace('_', ' ').title(),
            yaxis_title="Count",
            barmode='group',
            template="plotly_white",
        )
        figures[feature] = fig.to_html()
    
    return figures, feat_available

