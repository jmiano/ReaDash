import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from sklearn.preprocessing import OneHotEncoder, StandardScaler

import plotly.graph_objs as go
from plotly.offline import iplot
from sklearn.metrics import accuracy_score


def prepare_data(data, OH, sscaler):
    # Do OneHot Encoding
    # List of columns that we do not need to one hot encode
    no_OH = ['encounter_id', 'patient_nbr', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient',
             'number_emergency', 'number_inpatient', 'number_diagnoses', 'medical_specialty', 'payer_code', 'readmitted']

    X_continuous = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient',
             'number_emergency', 'number_inpatient', 'number_diagnoses']


    # Do OneHot encoding of remaining columns
    X_OH = data.drop(no_OH, axis=1)
    X_OH = OH.transform(X_OH)
    X_OH_df = pd.DataFrame(X_OH.toarray(), columns=OH.get_feature_names())
    
    # Feature scaling
    X_normed = sscaler.transform(data[X_continuous])
    X_normed = pd.DataFrame(X_normed, columns=X_continuous)

    # Final df
    final_df = pd.concat([X_OH_df, X_normed], axis=1)
    final_df['readmitted'] = data['readmitted'].apply(lambda val: 0 if val=='NO' else 1)
    
    return final_df


def count_predictions(data, model):
    predictions = model.predict(data.loc[:, data.columns != 'readmitted'])
    predictions = pd.DataFrame(predictions)
    predictions.columns = ['Predictions']
    prediction_counts = pd.DataFrame(predictions.Predictions.value_counts()).reset_index()
    prediction_counts.columns = ['Class', 'Percent']
    prediction_counts['Class'] = (prediction_counts['Class']
                                        .apply(lambda row: 'Readmitted' if row == 1 else 'Not Readmitted'))
    return prediction_counts


def get_predictions(data, model):
    predictions = model.predict(data.loc[:, data.columns != 'readmitted'])
    predictions = pd.DataFrame(predictions)
    predictions.columns = ['Predictions']

    return predictions


def plot_predictions(data):
    
    # Make plots
    pie = go.Pie(labels=data.Class.values,
                 values=data.Percent.values,
                 opacity=1,
                 hole=0.4,
                 marker=dict(colors=["#007dcc", "#ba0000"]))
    
    layout = go.Layout(title='Model Prediction Distribution',
                       yaxis=dict(title='Percent', tickformat='%', range=(0,1)),
                       paper_bgcolor='rgba(0,0,0,0)')#,
                       #plot_bgcolor='rgba(0,0,0,0)')

    fig = go.Figure(data=[pie], layout=layout)
    with open("../Figures/PredictionsPlot.json", 'w') as f:
        f.write(fig.to_json())

    return fig


def generate_predictions(final_df):
    final_df = prepare_data(data, OH_Encoder, scaler)
    prediction_counts = count_predictions(final_df, model)
    plot_predictions(prediction_counts)


def make_hist(data):
    data['readmitted'] = data['readmitted'].apply(lambda row: 'Readmitted' if row in ['>30', '<30'] else 'Not Readmitted')
    columns = list(data.columns[2:])
    bars = []
    flag = True

    for col_name in columns:
        counts = pd.DataFrame(data[col_name].value_counts(normalize=True)).reset_index()
        counts.columns = [col_name, 'percent']

        # Make plots

        # Make first plot visible
        if flag:
            bar = go.Bar(x=list(counts[col_name]),
                         y=list(counts['percent']),
                         textposition="outside",
                         texttemplate="%{y:%}",
                         marker=dict(color="#007dcc", line=dict(color="#002f4c", width=0)),
                         name=col_name,
                         opacity=1,
                         visible=True
                         )

            flag = False  # deactivate flag

        # Create hidden subsequent plots
        else:
            bar = go.Bar(x=list(counts[col_name]),
                         y=list(counts['percent']),
                         textposition="outside",
                         texttemplate="%{y:%}",
                         marker=dict(color="#007dcc", line=dict(color="#002f4c", width=0)),
                         name=col_name,
                         opacity=1,
                         visible=False
                         )

        bars.append(bar)

    buttons = []

    for i, col_name in enumerate(columns):
        visible_vector = [False] * len(columns)
        visible_vector[i] = True

        plot_dict = dict(label=col_name,
                         method='update',
                         args=[{'visible': visible_vector},
                               {'title': f"Feature Distribution for {col_name.upper()}"}])

        buttons.append(plot_dict)

    # Menus
    updatemenus = [dict(active=0,
                        buttons=buttons,
                        pad={"r": 20},
                        showactive=True)]

    layout = go.Layout(title="Pick a column to plot its distribution",
                       updatemenus=updatemenus,
                       yaxis=dict(title='Percent', tickformat='%', range=(0, 1.1)),
                       paper_bgcolor='rgba(0,0,0,0)')#,
                       #plot_bgcolor='rgba(0,0,0,0)')

    fig = go.Figure(data=bars, layout=layout)
    with open("../Figures/HistPlot.json", 'w') as f:
        f.write(fig.to_json())

    return fig


def make_pca_plot(data, OH):
    no_OH = ['encounter_id', 'patient_nbr', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient',
             'number_emergency', 'number_inpatient', 'number_diagnoses', 'medical_specialty', 'payer_code', 'readmitted']

    X_continuous = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient',
             'number_emergency', 'number_inpatient', 'number_diagnoses']

    X_OH = data.drop(no_OH, axis=1)
    X_OH = OH.transform(X_OH)
    X_OH_df = pd.DataFrame(X_OH.toarray(), columns=OH.get_feature_names())

    sscaler = StandardScaler().fit(data[X_continuous])
    X_normed = sscaler.transform(data[X_continuous])
    X_normed = pd.DataFrame(X_normed, columns=X_continuous)
    final_df = pd.concat([X_OH_df, X_normed], axis=1)

    # Apply PCA
    pca = PCA(n_components=3)
    pca.fit(final_df)
    PCA_df = pd.DataFrame(pca.transform(final_df), columns=['pc1', 'pc2', 'pc3'])

    PCA_df['labels'] = data['readmitted'].apply(lambda val: False if val=='NO' else True) #relabel multiclass to be binary

    PCA_df_ss = PCA_df.sample(50_000) #downsample


    # Make plots
    scatter_1 = go.Scatter3d(x=PCA_df_ss[PCA_df_ss.labels].pc1.values,
                             y=PCA_df_ss[PCA_df_ss.labels].pc2.values,
                             z=PCA_df_ss[PCA_df_ss.labels].pc3.values,
                             name="Readmitted",
                             mode ='markers',
                             marker=dict(size=2,
                                         opacity = 1,
                                         color="#ba0000"))

    scatter_0 = go.Scatter3d(x=PCA_df_ss[~PCA_df_ss.labels].pc1.values,
                             y=PCA_df_ss[~PCA_df_ss.labels].pc2.values,
                             z=PCA_df_ss[~PCA_df_ss.labels].pc3.values,
                             name="Not Readmitted",
                             mode ='markers',
                             marker=dict(size=2,
                                         opacity = 1,
                                         color = "#007dcc"))

    layout = go.Layout(title='Plot of data in PCA space',
                        scene=dict(xaxis=dict(range=[-6,8]),
                                   yaxis=dict(range=[-2,42]),
                                   zaxis=dict(range=[-2,27])),
                       scene_aspectmode='cube',
                       scene_camera=dict(eye=dict(x=1, y=1, z=2)),
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)')

    fig = go.Figure(data=[scatter_1, scatter_0], layout=layout)
    with open("../Figures/PCAPlot.json", 'w') as f:
        f.write(fig.to_json())

    return fig



def plot_feature_importances(final_df, model):
    fi = pd.DataFrame(sorted(zip(final_df.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)[:20])
    fi.columns = ['feature', 'importance']

    # Make plots
    bar = go.Bar(x=fi.feature.values, 
                 y=fi.importance.values,
                 opacity=1,
                 marker=dict(color="#007dcc", line=dict(color="#002f4c", width=0))
                )

    layout = go.Layout(title='Random Forest Feature Importances',
                       xaxis=dict(tickangle=45),
                       yaxis=dict(title='Feature Importance'),
                       paper_bgcolor='rgba(0,0,0,0)')#,
                       #plot_bgcolor='rgba(0,0,0,0)')

    fig = go.Figure(data=[bar], layout=layout)
    with open("../Figures/FeatureImportances.json", 'w') as f:
        f.write(fig.to_json())

    return fig


def plot_performance(X_test, y_test, models, metrics):    
    df_performance_metrics=[]

    for model in models:
        row = []
        row.append(type(model[0]).__name__)
        
        loaded_model = pickle.load(open('../Models/{}_model.pkl'.format(model[1]), 'rb'))
        predictions = loaded_model.predict(X_test)

        row.append(accuracy_score(y_true=y_test, y_pred=predictions))
        
        for metric in metrics:
            row.append(metric(y_true=y_test, y_pred=predictions, average='macro'))
            
        df_performance_metrics.append(row)


    df_performance_metrics = pd.DataFrame(df_performance_metrics,
                                            columns = ['model', 'accuracy_score', 'f1_score', 'precision_score', 'recall_score'])
    df_performance_metrics_melt = pd.melt(df_performance_metrics, 
                                            id_vars=['model'], 
                                            value_vars=['accuracy_score', 'f1_score', 'precision_score', 'recall_score'])

    bars = []

    for model in df_performance_metrics_melt.model.unique():
        temp_df = df_performance_metrics_melt[df_performance_metrics_melt.model == model]
        
        bar = go.Bar(x=temp_df.variable.values, 
                     y=temp_df.value.values,
                     opacity=1,
                     textposition = "outside",
                     #texttemplate = "%{y:%}",
                     name=model
                    )
        
        bars.append(bar)

    layout = go.Layout(title='Performance Metrics',
                       yaxis=dict(title='Percent', tickformat='%', range=(0,1.1)),
                       legend=dict(orientation="h", 
                                   y=-0.1, x=0.5,
                                   yanchor='top', xanchor='center'),
                       paper_bgcolor='rgba(0,0,0,0)')#,
                       #plot_bgcolor='rgba(0,0,0,0)')

    fig = go.Figure(data=bars, layout=layout)

    with open("../Figures/PerformanceMetrics.json", 'w') as f:
        f.write(fig.to_json())

    return fig