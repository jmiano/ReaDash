import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import json
import pandas as pd
import base64
import io
from components import prepare_data, get_predictions, count_predictions, plot_predictions
import pickle


# Style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Dash App
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, prevent_initial_callbacks=True)

# Load Figures
with open('../Figures/HistPlot.json', 'rb') as f:
    fig_histograms = json.load(f)
with open('../Figures/PredictionsPlot.json', 'rb') as f:
    fig_predictions = json.load(f)
with open('../Figures/PCAPlot.json', 'rb') as f:
    fig_PCA = json.load(f)
with open('../Figures/FeatureImportances.json', 'rb') as f:
    fig_feature_importances = json.load(f)
with open('../Figures/PerformanceMetrics.json', 'rb') as f:
    fig_performance = json.load(f)

app.layout = html.Div(children=[
    html.Center(html.H1(children='Hospital Readmissions for Diabetes Patients')),
    html.Hr(style={'border':'1px solid #5fa8e0'}),  #create horizontal line


    html.Center(children=[html.H6(children='Dashboard showing exploratory analyses and results from training machine '
                                           'learning models to predict hospital readmissions in diabetes patients. '
                                           'The user can input data for which they want to generate outputs from a '
                                           'random forest model.',
                                    style={'width': '50%',
                                           'display': 'inline-block',
                                           'vertical-align': 'middle',
                                           'margin-bottom': 20,
                                           'margin-top': -10,
                                           'text-align': 'center'}),
                  dcc.Upload(
                      id='upload_csv',
                      children=[
                          html.Button('Use Model',
                                      style={
                                          'display': 'inline-block',
                                          'vertical-align': 'middle',
                                          'margin': 2,
                                          'width': '8%',
                                          'border-radius': '5px',
                                          'background-color': "#007dcc",
                                          'color': 'white'
                                      }
                                      )
                      ]
                  ),
                html.Center([html.Button('Download Outputs', id='download_outputs_btn', n_clicks=0,
                            style={
                                'display': 'inline-block',
                                'vertical-align': 'middle',
                                'margin': 2,
                                'width': '8%',
                                'border-radius': '5px',
                                'background-color': "#007dcc",
                                'color': 'white'
                            }),
                dcc.Download(id="download_csv")]),]),
    html.Center(children=[
        dcc.Graph(
            id='histograms',
            figure=fig_histograms,
            style={
                'width':'40%',
                'display':'inline-block',
                'border':'2px solid #5fa8e0',
                'border-radius': '20px',
                'margin':20,
                'vertical-align': 'middle'
            }
        ),
        html.U(id='performance_outputs')]),
    #     dcc.Graph(
    #         id='predictions',
    #         figure=fig_predictions,
    #         style={
    #             'width': '40%',
    #             'display': 'inline-block',
    #             'border': '2px solid #5fa8e0',
    #             'border-radius': '20px',
    #             'margin': 20,
    #             'vertical-align': 'middle'
    #         }
    # )]),


    html.Center(children=[
        dcc.Graph(
            id='pca',
            figure=fig_PCA,
            style={
                'width':'30%',
                'display':'inline-block',
                'border':'2px solid #5fa8e0',
                'border-radius': '20px',
                'margin':20,
                'vertical-align': 'middle'
            }
        ),

        dcc.Graph(
            id='performance',
            figure=fig_performance,
            style={
                'width': '30%',
                'display': 'inline-block',
                'border': '2px solid #5fa8e0',
                'border-radius': '20px',
                'margin': 20,
                'vertical-align': 'middle'
            }
        ),

        dcc.Graph(
            id='feature_importances',
            figure=fig_feature_importances,
            style={
                'width':'30%',
                'display':'inline-block',
                'border':'2px solid #5fa8e0',
                'border-radius': '20px',
                'margin':20,
                'vertical-align': 'middle'
            }
        )])



])

def parse_contents(contents):
    c_type, c_string = contents.split(',')

    decoded = base64.b64decode(c_string)
    try:
        data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return html.Div([
            'Error uploading this file. Please upload a CSV file only.'
        ])
    return data

@app.callback(
    Output("download_csv", "data"),
    Input("download_outputs_btn", "n_clicks"),
    prevent_initial_call=True
)
def download(n_clicks):
    df = pd.read_csv('predictions.csv')
    return dcc.send_data_frame(df.to_csv, "predictions.csv")

@app.callback(Output('performance_outputs', 'children'),
    Input('upload_csv', 'contents'),
    State('upload_csv', 'filename')
)
def upload(content, name):
    if content is not None:
        data = parse_contents(content)

        model = pickle.load(open('../Models/rf_model.pkl', 'rb'))
        scaler = pickle.load(open('../Models/scaler.pkl', 'rb'))
        OH = pickle.load(open('../Models/OH_Encoder.pkl', 'rb'))

        final_df = prepare_data(data, OH, scaler)
        predictions = get_predictions(final_df, model)
        predictions.to_csv("predictions.csv", index=False)

        count_preds = count_predictions(final_df[final_df.columns[:-1]], model)
        fig = plot_predictions(count_preds)

        plot = dcc.Graph(
            id='predictions',
            figure=fig,
            style={
                'width': '40%',
                'display': 'inline-block',
                'border': '2px solid #5fa8e0',
                'border-radius': '20px',
                'margin': 20,
                'vertical-align': 'middle'
                }
            )
        return plot

if __name__ == '__main__':
    app.run_server(debug=True)