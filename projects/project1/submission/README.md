#Groupe 1 : ML_project1

Project from: Loïs Huguenin : 234887 <br />
Julie Djeffal : 193164 <br />
Fabien Zellweger : 209450 <br />

## Project

- The run.py file is used to generate the csv file that gives us our best results.
- implementations.py refer to the 6 methods we were asked to provide.
- csv folder. Empty because the train.csv and test.csv are too big for us. You have to add they in this folder
- Methods folder contains all the helper *.py file used by run.py and all our tests.

# Run.py WARNING !!!!
- run.py gives our best predictions on Ubuntu 14.04 x64 kernel v.3.16 but apparently have error on Windows and MacOSx (singular hessian matrix)

## Mandatory Methods Warning
- For least_squares_GD and least_squares_SGD the init weights MUST be a 1D vector

## Boxplots
- The boxplots are plotted by the boxplot method from plots.py

## Data Preparation
- To scale the data, the method data_scaling in scaling_standardization.py is used
- For the simpler models, we simply used the build_poly_matrix method in build_poly.py
- For our best ML model, we loop on the column of tX and use the add_feature method in build_poly.py (see run.py for example)

## Model analysis
- All RMSE are computed with the error e = y - prediction(dot(X, w_train)) and not e = y - dot(X, w_train)
- Before logistic regression, all -1 in y are replaced by 0 and the inverse operation is done once we have the predictions
- One step of CV is performed by the cross_validation method in cross_validation.py, hence to compute complete CV you must compute:
  k_indices = build_k_indices(y, k_fold, seed)
  for k in k_fold
      loss_tr, loss_te, accuracy, w_tr = cross_validation(y, x, k_indices, k, model, parameters for the model)

  Where k_indices is a method in cross_validation.py

## Best Model
- The best model is computed on 70% of transformed data, we use split_data(x, y_binary, 0.7) which is a method in split_data.py
- Then logistic regression with newton's method and mini-batch of 3000 entries is performed, max_iters = 750, gamma = 0.005, for that we use logistic_regression_n from logistic.py

## How to execute the code

- Add the train.csv and test.csv in the 'csv' folder
- Run run.py
- Wait some time, the transformations take time and the logistic regression takes 750 steps (few minutes on our computers)
- The predictions are put in the 'csv' folder

## Helper methods

- build_polynomial.py Contains methods for features transformations
- clearDataset.py Contains methods to remove -999 (used to try model with this data cleaning method)
- costs.py Methods to compute mse, mae, ...
- cross_validation.py Methods for cross validation
- helpers.py Some useful methods
- logistic Logistic regression with newton's method
- plots.py Boxplots
- proj1_helpers.py Load, generate csv
- scaling_standardization.py Methods to scale and standardize data
- split_data Methods to split the data randomly
