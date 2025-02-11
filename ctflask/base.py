from flask import Flask, render_template
import dash
from dash import dcc, html

# Initialize Flask
server = Flask(__name__)

# Initialize Dash inside Flask
dash_app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

# Dash Layout
dash_app.layout = html.Div([
    html.H1("Dash Dashboard"),
    dcc.Graph(
        figure={"data": [{"x": [1, 2, 3], "y": [4, 1, 2], "type": "line"}]}
    )
])


# Flask Route for Home Page
@server.route('/')
def home():
    return render_template('index.html')  # Load HTML page with the button


# Run Flask App
if __name__ == '__main__':
    server.run(debug=True)
