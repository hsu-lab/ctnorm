import argparse
from flask import Flask, render_template, redirect, url_for, request, jsonify, flash, session
import os
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from .helpers import *
import sys
import pandas as pd
from dash.dependencies import Input, Output, State
import nibabel as nib
from dash_slicer import VolumeSlicer
from ctnorm.Harmonization.data.utils import read_data
import datetime
import json


app = Flask(__name__, template_folder="templates")
app.secret_key = os.urandom(24)

external_stylesheets = [dbc.themes.BOOTSTRAP]
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dicom/', external_stylesheets=external_stylesheets)
dash_app.layout = html.Div(id="dynamic-content")


@app.route("/reset", methods=["GET"])
def reset_sess():
    session.pop("user", None)
    return redirect(url_for("home"))  # Redirect back to home if accessed via GET


@app.route("/")
def home():
    session_list = []
    session_base_path = app.config.get("SESSION_FOLDER")  # Default folder
    for session_folder in os.listdir(session_base_path):
        session_path = os.path.join(session_base_path, session_folder)
        session_status_file = os.path.join(session_path, "session_status.json")
        if os.path.isdir(session_path) and os.path.exists(session_status_file):
            try:
                # Read session_status.json
                with open(session_status_file, "r") as f:
                    session_data_json = json.load(f)
                session_data = {
                    "name": session_data_json.get("session_id", session_folder),  # Session ID or folder name
                    "status": session_data_json.get("status", "Unknown"),
                    "timestamp": session_data_json.get("timestamp", "N/A"),
                    "failed_modules": [],
                }
                # If session failed, collect all failed modules & their error messages
                if session_data["status"] == "failed":
                    failed_modules = session_data_json.get("module_status", {})
                    error_messages = session_data_json.get("error_messages", {})

                    for module, status in failed_modules.items():
                        if status == "failed":
                            session_data["failed_modules"].append({
                                "module": module,
                                "error": error_messages.get(module, "No details available")
                            })

                session_list.append(session_data)
            except Exception as e:
                print(f"Error reading {session_status_file}: {e}")
    return render_template("home.html", session_list=session_list)


@app.route("/load_session-c", methods=["GET", "POST"])
def load_sess():
    if request.method == "POST":
        session_number = request.form["session_number"]
        session_base_path = app.config.get("SESSION_FOLDER")
        if session_number not in os.listdir(session_base_path):
            flash("Error: Session does not exist", "error")
            return redirect(url_for("home"))  # Redirect to home with error message
        else:
            session["user"] = session_number
            return load_char() # Load Characterization module by default
    else:
        # if session.get("user"):
        #     return render_template("session_characterization.html", sess=session["user"])
        return redirect(url_for("home"))  # Redirect back to home if accessed via GET


@app.route("/load_char-p", methods=["GET", "POST"])
def load_char():
    if session.get("user"):
        global avail_feat # Declare this as a session variable to be accessed by other function 
        # Load the harmonization visualization for now (should go to characterization first)
        session_base_path = app.config.get("SESSION_FOLDER")
        INFO = os.path.join(session_base_path, session["user"])
        figures, avail_feat = plot_characterization(INFO)
        rad_key = next(iter(avail_feat), None)  # Returns None if dictionary is empty
        if rad_key:
            rad_key = avail_feat[rad_key]['feat_names']
        return render_template(
            "session_characterization.html",
            active_c=True,
            sess=session["user"],
            figures=figures,
            param={'feature_names': rad_key}
        )
    return redirect(url_for("home"))  # Redirect back to home if accessed via GET


@app.route('/submit-dataset', methods=['GET', 'POST'])
def handle_dataset_submission():
    if request.method == "POST" and session.get("user"):
        data = request.get_json()
        dataset_value = data.get('dataset')
        session_base_path = app.config.get("SESSION_FOLDER")
        try:
            f_path = os.path.join(session_base_path, session["user"], 'Harmonization', dataset_value, 'test')
            filenames = os.listdir(f_path)
            if len(filenames) > 0:
                metrics = [f for f in os.listdir(os.path.join(f_path, filenames[0])) if 'metadata' not in f]
                models = [f.split('--')[0] for f in os.listdir(os.path.join(f_path, filenames[0], metrics[0]))]
            return jsonify({
                "status": "success",
                "files": filenames,
                "metrics": metrics,
                "models": models,
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            })
    else:
        return redirect(url_for("home"))


@app.route("/load_session-p", methods=["GET", "POST"])
def load_preprocessing():
    if session.get("user"):
        # Load the harmonization visualization for now (should go to characterization first)
        session_base_path = app.config.get("SESSION_FOLDER")
        INFO = os.path.join(session_base_path, session["user"], 'Harmonization')
        available_d = os.listdir(INFO)
        return render_template("session_preprocessing.html", active_p=True, datasets=available_d, sess=session["user"])
    return redirect(url_for("home"))  # Redirect back to home if accessed via GET


@app.route('/submit-preprocessing', methods=['POST'])
def handle_preprocessing():
    if request.method == "POST" and session.get("user"):
        session_base_path = app.config.get("SESSION_FOLDER")
        data = request.get_json()
        datasetid, caseid, metrics, models = data.get('datasetid'), data.get('caseid'), data.get('metrics'), data.get('models')

        cases_pth = os.path.join(session_base_path, session["user"], 'Harmonization', datasetid, 'test', caseid, 'Volume')
        filter_cases = [os.path.join(cases_pth, f"{mod}--{caseid}") for mod in models]
        slicers = []
        for case in filter_cases:
            slicer_name = case.split('/')[-1].split('--')[0]  # Extract name
            _, _, img = read_data(case, ext='dcm', apply_lut_for_dcm=False)

            slicer = VolumeSlicer(dash_app, img, axis=0, thumbnail=False)
            slicer.graph.figure.update_layout(plot_bgcolor="rgb(0, 0, 0)") 
            slicer.graph.config.update(modeBarButtonsToAdd=[])
            slicer.slider.marks = None  
            slicer.slider.tooltip = {"always_visible": False}  
            slicers.append((slicer_name, slicer))

        # Create Cards (3 per row)
        cards = [
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(name, className="text-center"),  # Centered title
                        dbc.CardBody([
                            slicer.graph,  
                            html.Div(style={"margin-top": "15px"}), 
                            dbc.Row(slicer.slider, className="mt-3"),
                            *slicer.stores  # Stores for interactivity
                        ]),
                    ]), width=4
                ) for name, slicer in slicers[i:i+3]
            ], className="mb-4")  
            for i in range(0, len(slicers), 3)
        ]

        # Update Dash Layout Dynamically
        dash_app.layout = html.Div(dbc.Container(cards, fluid=True))
        return jsonify({'status': 'success', 'message': 'Dashboard updated successfully'})


@app.route("/read-feature-multiple", methods=["POST"])
def readfeature_multiple():
    if request.method == "POST" and session.get("user"):
        data = request.get_json()
        feature_names = data.get('feature_names', [])  # Already cleaned feature names
        print('Reading features:', feature_names)

        if (not avail_feat) or (len(feature_names) == 0):
            return jsonify({'status': 'failed', 'message': 'Radiomic features cannot be loaded!'})
        figures = plot_radiomics(avail_feat, feature_names)
        return jsonify({'status': 'success', 'plots': figures})
    else:
        return redirect(url_for("home"))


def run_server():
    """Launch the Flask server with basic validation."""
    parser = argparse.ArgumentParser(description="Run CTNorm Flask WebApp")
    parser.add_argument("--port", type=int, required=True, help="Port number to run Flask on")
    parser.add_argument("--session-out", type=str, required=True, help="Path to session folder for visualization")
    args = parser.parse_args()

    # Ensure session folder exists and is not empty
    if not os.path.isdir(args.session_out) or not os.listdir(args.session_out):
        print(f"ERROR: Session folder '{args.session_out}' does not exist or is empty.")
        sys.exit(1)

    # Store session folder path globally in Flask
    app.config["SESSION_FOLDER"] = args.session_out

    print(f"🚀 Starting CTNorm WebApp on port {args.port}")
    print(f"📂 Using session folder: {args.session_out}")

    app.run(host="0.0.0.0", port=args.port, debug=True)

