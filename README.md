# ReaDash
Author: Joseph Miano

## Summary
Repo for my CS 6440 Final Practicum Project Deliverables.

ReaDash is an interactive dashboard displaying information relating to diabetes patient hospital readmissions. The user can visualize plots related to features predictive of hospital readmission, interact with a machine learning model (logistic regression) in order to download predictions for their own dataset, and visualize the performance of various machine learning models in predicting hospital readmission for patients with diabetes.

# Application Manual

### Instructions for Dashboard Access
Navigate to http://18.216.177.129:9001/ in a web browser to access the dashboard.

### Instructions for Dashboard Use
All plots in the dashboard are interactive and can be clicked to zoom or highlight data.

To upload your own data and get predictions from a logistic regression model, use the "Use Model" button.
To download the predictions from the model for your data, use the "Download Outputs" button (after using "Use Model").

Note: your data must be in the correct format to use the model to get predictions.
Please visit https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008
and download the dataset to see the correct format and all the features needed.


### Dashboard Visualization Information
When you first access the dashboard, you will see a title, description, the 2 buttons, and 4 plots (1 on the top row and 3 on the bottom row).

The top plot displays the distribution of values (in the overall dataset linked above) for the selected feature, which can be chosen with the dropdown menu.

The bottom left plot displays the overall dataset in a 3-dimensional space after applying PCA, separated by readmitted and not readmitted patients.

The bottom middle plot displays the performance metrics for the various machine learning models we tested on our validation dataset.

The bottom right plot displays the feature importances in decreasing order for our random forest classifier.

Once you use the "Use Model" button, a new plot will appear on the right of the first row displaying the percent of predictions from the model for each class.
The "Use Model" and "Download Outputs" buttons can be used as many times as needed, and the "Use Model" button will continue to update the top right figure.
