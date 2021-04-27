import pandas as pd
import pickle
from components import (prepare_data, count_predictions, plot_predictions, generate_predictions,
    make_hist, make_pca_plot, plot_feature_importances, plot_performance)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, precision_score, recall_score

# Prepare Data
data = pd.read_csv('../Datasets/diabetic_data.csv')
X_test = pd.read_csv('../Datasets/X_test.csv')
y_test = pd.read_csv('../Datasets/y_test.csv')
model = pickle.load(open('../Models/rf_model.pkl', 'rb'))
scaler = pickle.load(open('../Models/scaler.pkl', 'rb'))
OH = pickle.load(open('../Models/OH_Encoder.pkl', 'rb'))
final_df = prepare_data(data, OH, scaler)
prediction_counts = count_predictions(final_df, model)
models = [(RandomForestClassifier(n_jobs=6), 'rf'),
          (AdaBoostClassifier(), 'adaboost'),
          (LogisticRegression(), 'log_reg'),
          (MLPClassifier(hidden_layer_sizes=(10,)), 'nn'),
          (GaussianNB(), 'gnb'),
          (DecisionTreeClassifier(), 'dtree')]

metrics = [f1_score, precision_score, recall_score]

# Get figures
plot_predictions(prediction_counts)
make_pca_plot(data, OH)
plot_feature_importances(final_df, model)
plot_performance(X_test[X_test.columns[1:]], y_test['readmitted'], models, metrics)
make_hist(data)