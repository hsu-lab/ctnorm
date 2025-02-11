import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


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
                csv_path = os.path.join(dataset_path, "data_characterization_dummy.csv")
                rad_path = os.path.join(dataset_path, "rad_feat.csv")

                if os.path.exists(rad_path):
                    feature_names = [f.split('_')[-1] for f in pd.read_csv(rad_path).columns.tolist()[1:]]  # Exclude the first column (e.g., 'Dataset')
                    feat_available[dataset] = {'rad_path': rad_path, 'feat_names': feature_names}
                if os.path.exists(hist_path) and os.path.exists(csv_path):
                    histogram_data[dataset] = np.load(hist_path)
                    csv_data[dataset] = pd.read_csv(csv_path)

                else:
                    continue  # Skip this dataset if required files are missing
            else:
                continue
    else:
        return figures, feat_available  # Return empty if module path doesn't exist

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
        if len(set(hist_data)) > 1:  # Ensure valid KDE computation
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
    
    # **Updated SNR Plot (Box Plot)**
    fig_snr = go.Figure()
    for idx, (dataset, df) in enumerate(csv_data.items()):
        if 'snr' in df.columns:
            fig_snr.add_trace(go.Box(
                y=df['snr'],
                name=dataset,
                marker=dict(color=colors[idx % len(colors)]),
                boxpoints="all",  # Show all individual points
                jitter=0.3,  # Spread out individual points
                pointpos=-1.8  # Offset points to avoid overlap
            ))

    fig_snr.update_layout(
        title="Signal-to-Noise Ratio (SNR) Distribution",
        yaxis_title="SNR",
        xaxis_title="Datasets",
        template="plotly_white",
    )
    figures['snr'] = fig_snr.to_html()

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


def plot_radiomics(avail_feat, feature_names):
    figures = {}  # Dictionary to store scatter plots
    dataset_features = {}  # Store feature data for each dataset
    colors = ["blue", "pink", "green", "purple", "orange", "red", "cyan", "magenta"]
    # **Step 1: Load and Process Datasets Once**
    for dataset in avail_feat:
        df = pd.read_csv(avail_feat[dataset]['rad_path'])

        # Create a mapping: {clean_feature_name: full_column_name}
        column_mapping = {col.split('_')[-1]: col for col in df.columns}

        # Store only the relevant feature columns
        dataset_features[dataset] = {
            feature: df[column_mapping[feature]] for feature in feature_names if feature in column_mapping
        }

    # **Step 2: Generate Scatter Plots for Each Feature**
    for feature in feature_names:
        fig = go.Figure()

        for idx, dataset in enumerate(dataset_features):
            if feature in dataset_features[dataset]:
                fig.add_trace(go.Scatter(
                    x=dataset_features[dataset][feature].index,  # Use index as X-axis
                    y=dataset_features[dataset][feature],  # Feature values as Y-axis
                    mode="markers",
                    name=dataset,  # Dataset name in legend
                    marker=dict(color=f"rgba({idx * 30}, {255 - idx * 40}, {idx * 50}, 0.8)"),
                    marker_color=colors[idx % len(colors)]
                ))

        # Update figure layout
        fig.update_layout(
            title=f"{feature}",
            xaxis_title="Index",
            yaxis_title=feature,
            template="plotly_white"
        )
        # Store the figure in HTML format
        figures[feature] = fig.to_html()
    return figures