# import os
# import pandas as pd
# import numpy as np
# import glob
# from sybil import Sybil
# from sybil import Serie, visualize_attentions  # Adjust the import based on actual module names
# import torch
# from argparse import Namespace
# from sybil.utils.metrics import get_survival_metrics

# def process_cases(dataset, data_csv_path, output_dir, global_logger, metadata, MAX_FOLLOWUP= 6, num_years=6  ):
#     # Set the number of threads for torch
#     output_path = os.path.join(output_dir, 'risk_scores.csv')
#     vis_dir_path = os.path.join(output_dir, 'attention_maps')
#     num_threads = os.cpu_count() // 5
#     # Initialize the Sybil model
#     model = Sybil("sybil_ensemble")
#     # Load the CSV file
#     df = pd.read_csv(data_csv_path)
#     # df = df[:2]
#     # Drop rows where 'Directory' is NaN
#     df = df.dropna(subset=['Directory'])
#     global_logger.info(f"Number of non-null entries in 'Directory': {len(df)}")
#     # Add columns for predicted risk scores if they don't already exist
#     for i in range(num_years):
#         if f'pred_risk_year_{i}' not in df.columns:
#             df[f'pred_risk_year_{i}'] = np.nan
#     # Iterate through each row in the df
#     for index, row in df.iterrows():
#         global_logger.info(f'Processing case {index}')
#         # Get the DICOM directory, event, and years to event
#         dicom_dir = row['Directory']
#         event = row['event']
#         years_to_event = row['years_to_event']
#         pid = row['pid']
#         dicom_list = glob.glob(dicom_dir + '/*')
#         serie = Serie(dicom_list, label=event, censor_time=years_to_event)
#         results = model.predict([serie], return_attentions=True, threads=num_threads)
#         # Update the risk scores columns for the current row
#         for i in range(num_years):
#             df.at[index, f'pred_risk_year_{i}'] = results.scores[0][i]
#         global_logger.info(f"Risk scores: {results.scores[0]}")
#         # Save attention maps
#         attentions = results.attentions
#         visualize_attentions(
#             serie,
#             attentions=attentions,
#             pid=pid,
#             save_directory=vis_dir_path,
#             gain=1
#         )
#     # Save the results to the output CSV file
#     df.to_csv(output_path, mode='w', header=True, index=False)
#     global_logger.info(f"Processed all cases. Results saved to {output_path}.")
#     for metadata_column in metadata:
#         if metadata_column not in df.columns:
#             raise ValueError(f"Column '{metadata_column}' not found in dataset. Available columns: {list(df.columns)}")
#         # Get unique values of the metadata column with counts
#         metadata_counts = df[metadata_column].value_counts()
#         valid_metadata_values = metadata_counts[metadata_counts >= 10].index.tolist()
#         print(f"Valid {metadata_column} values (≥10 cases): {valid_metadata_values}")
#         # Process each valid metadata value separately
#         for metadata_value in valid_metadata_values:
#             print(f"Processing for {metadata_column}: {metadata_value}")
#             # Filter dataset for the current metadata value
#             subset_df = df[df[metadata_column] == metadata_value].copy()
#             # Generate per-year labels
#             for i in range(MAX_FOLLOWUP):
#                 subset_df[f'y_{i}'] = np.nan
#             def calculate_y_seq(event, time_to_event, max_followup=6):
#                 """Calculate binary outcome sequence based on event and time_to_event."""
#                 y_seq = np.zeros(max_followup)
#                 if event == 1:
#                     event_year = int(time_to_event)
#                     event_year = min(event_year, max_followup)
#                     y_seq[event_year:] = 1
#                 return y_seq
#             # Iterate through subset and compute labels
#             for index, row in subset_df.iterrows():
#                 y_seq = calculate_y_seq(row['event'], row['years_to_event'], MAX_FOLLOWUP)
#                 for i in range(MAX_FOLLOWUP):
#                     subset_df.at[index, f'y_{i}'] = y_seq[i]
#             # Print statistics
#             year0_diagnosis = np.count_nonzero((subset_df['y_0'] == 1))
#             total = len(subset_df)
#             print(f'Total cases for {metadata_value} = {total}')
#             print(f'Baseline lung cancer diagnoses = {year0_diagnosis} ({100 * year0_diagnosis / total:.2f}%)')
#             # Prepare data for survival metrics calculation
#             selected_columns = [f'pred_risk_year_{i}' for i in range(MAX_FOLLOWUP)]
#             pred_risk_scores = subset_df[selected_columns].values.tolist()
#             event_times = subset_df['years_to_event'].tolist()
#             event_observed = subset_df['event'].tolist()
#             input_dict = {
#                 "probs": torch.tensor(pred_risk_scores),
#                 "censors": torch.tensor(event_times),
#                 "golds": torch.tensor(event_observed),
#             }
#             args = Namespace(
#                 max_followup=MAX_FOLLOWUP, censoring_distribution=model._censoring_dist
#             )
#             # Compute survival metrics
#             out = get_survival_metrics(input_dict, args)
#             print(f"Results for {metadata_value}: {out}")
#             # Save the results
#             output_file = os.path.join(output_dir, f"{metadata_column}_{metadata_value}_results.csv")
#             subset_df.to_csv(output_file, index=False)
#             print(f"Results saved to {output_file}")
# # Example usage
# # nlst_with_labels_path = '/workspace/robustness_nlst_ip.csv'
# # output_path = '/workspace/risk_scores.csv'
# # vis_dir_path = "/workspace/attention_maps"
# # if not os.path.exists(vis_dir_path):
# #     os.makedirs(vis_dir_path)
# # process_cases(nlst_with_labels_path, output_path, vis_dir_path)

import os
import pandas as pd
import numpy as np
import glob
from sybil import Sybil
from sybil import Serie, visualize_attentions
import torch
from argparse import Namespace
from sybil.utils.metrics import get_survival_metrics
def process_cases(dataset, data_csv_path, output_dir, global_logger, metadata, MAX_FOLLOWUP=6, num_years=6):
    # Set the number of threads for torch
    output_path = os.path.join(output_dir, 'risk_scores.csv')
    vis_dir_path = os.path.join(output_dir, 'attention_maps')
    num_threads = max(1, os.cpu_count() // 5)  # Ensure at least one thread
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # Initialize the Sybil model
    try:
        model = Sybil("sybil_ensemble")
    except Exception as e:
        global_logger.error(f"Error initializing Sybil model: {e}")
        return
    # Load the CSV file
    if not os.path.exists(data_csv_path):
        global_logger.error(f"Data CSV file not found: {data_csv_path}")
        return
    try:
        df = pd.read_csv(data_csv_path)
    except Exception as e:
        global_logger.error(f"Error reading CSV file: {e}")
        return
    if 'Directory' not in df.columns:
        global_logger.error("Missing 'Directory' column in dataset.")
        return
    df = df.dropna(subset=['Directory'])
    global_logger.info(f"Number of non-null entries in 'Directory': {len(df)}")
    # Ensure predicted risk columns exist
    for i in range(num_years):
        col_name = f'pred_risk_year_{i}'
        if col_name not in df.columns:
            df[col_name] = np.nan
    # Iterate through each row in the df
    for index, row in df.iterrows():
        try:
            global_logger.info(f'Processing case {index}')
            # Check if required columns exist in the row
            required_cols = ['Directory', 'event', 'years_to_event', 'pid']
            if not all(col in row for col in required_cols):
                global_logger.warning(f"Skipping row {index}: Missing required columns")
                continue
            dicom_dir = row['Directory']
            event = row['event']
            years_to_event = row['years_to_event']
            pid = row['pid']
            if not os.path.exists(dicom_dir):
                global_logger.error(f"Skipping {index}: DICOM directory not found - {dicom_dir}")
                continue
            dicom_list = glob.glob(dicom_dir + '/*')
            if not dicom_list:
                global_logger.error(f"Skipping {index}: No DICOM files found in {dicom_dir}")
                continue
            serie = Serie(dicom_list, label=event, censor_time=years_to_event)
            results = model.predict([serie], return_attentions=True, threads=num_threads)
            # Update risk scores
            for i in range(num_years):
                df.at[index, f'pred_risk_year_{i}'] = results.scores[0][i]
            global_logger.info(f"Risk scores for {index}: {results.scores[0]}")
            # Save attention maps
            attentions = results.attentions
            visualize_attentions(
                serie,
                attentions=attentions,
                pid=pid,
                save_directory=vis_dir_path,
                gain=1
            )
        except Exception as e:
            global_logger.error(f"Error processing case {index}: {e}")
    # Save the results to the output CSV file
    try:
        df.to_csv(output_path, mode='w', header=True, index=False)
        global_logger.info(f"Processed all cases. Results saved to {output_path}.")
    except Exception as e:
        global_logger.error(f"Error saving results to CSV: {e}")
    # Process metadata
    for metadata_column in metadata:
        if metadata_column not in df.columns:
            global_logger.error(f"Column '{metadata_column}' not found in dataset. Available columns: {list(df.columns)}")
            continue
        metadata_counts = df[metadata_column].value_counts()
        valid_metadata_values = metadata_counts[metadata_counts >= 10].index.tolist()
        global_logger.info(f"Valid {metadata_column} values (≥10 cases): {valid_metadata_values}")
        for metadata_value in valid_metadata_values:
            try:
                global_logger.info(f"Processing for {metadata_column}: {metadata_value}")
                subset_df = df[df[metadata_column] == metadata_value].copy()
                # Generate per-year labels
                for i in range(MAX_FOLLOWUP):
                    subset_df[f'y_{i}'] = np.nan
                def calculate_y_seq(event, time_to_event, max_followup=6):
                    y_seq = np.zeros(max_followup)
                    if event == 1:
                        event_year = int(time_to_event)
                        event_year = min(event_year, max_followup)
                        y_seq[event_year:] = 1
                    return y_seq
                for index, row in subset_df.iterrows():
                    y_seq = calculate_y_seq(row['event'], row['years_to_event'], MAX_FOLLOWUP)
                    for i in range(MAX_FOLLOWUP):
                        subset_df.at[index, f'y_{i}'] = y_seq[i]
                year0_diagnosis = np.count_nonzero((subset_df['y_0'] == 1))
                total = len(subset_df)
                global_logger.info(f'Total cases for {metadata_value} = {total}')
                global_logger.info(f'Baseline lung cancer diagnoses = {year0_diagnosis} ({100 * year0_diagnosis / total:.2f}%)')
                # Prepare data for survival metrics calculation
                selected_columns = [f'pred_risk_year_{i}' for i in range(MAX_FOLLOWUP)]
                pred_risk_scores = subset_df[selected_columns].values.tolist()
                event_times = subset_df['years_to_event'].tolist()
                event_observed = subset_df['event'].tolist()
                input_dict = {
                    "probs": torch.tensor(pred_risk_scores),
                    "censors": torch.tensor(event_times),
                    "golds": torch.tensor(event_observed),
                }
                args = Namespace(
                    max_followup=MAX_FOLLOWUP, censoring_distribution=model._censoring_dist
                )
                # Compute survival metrics
                out = get_survival_metrics(input_dict, args)
                global_logger.info(f"Results for {metadata_value}: {out}")
                output_file = os.path.join(output_dir, f"{metadata_column}_{metadata_value}_results.csv")
                try:
                    subset_df.to_csv(output_file, index=False)
                    global_logger.info(f"Results saved to {output_file}")
                except Exception as e:
                    global_logger.error(f"Error saving metadata results to {output_file}: {e}")
            except Exception as e:
                global_logger.error(f"Error processing metadata {metadata_column} = {metadata_value}: {e}")






