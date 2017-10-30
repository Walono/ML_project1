import numpy as np
from Methods.costs import *
from Methods.split_data import *
from Methods.scaling_standardization import *
from Methods.features_processing import *
from Methods.logistic import *
from Methods.clearDataset import *

from Methods.proj1_helpers import *

print("Loading training data")
DATA_TRAIN_PATH = 'csv/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

print("Loading test data")
DATA_TEST_PATH = 'csv/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

print("Handling missing data by using averages")
tX, tX_test = averageData(tX, tX_test)
tX, tX_test = data_scaling(tX.T, tX_test.T)

print("Preparing the data by fitting features")
for i in range(0, 30):
    tX, tX_test = add_feature(y, tX[i], tX, log_def, tX_test, tX_test[i])
    tX, tX_test = add_feature(y, tX[i], tX, multiply, tX_test, tX_test[i], degree=2)
    tX, tX_test = add_feature(y, tX[i], tX, sqrt_def, tX_test, tX_test[i])
    tX, tX_test = add_feature(y, tX[i], tX, multiply, tX_test, tX_test[i], degree=3)
    tX, tX_test = add_feature(y, tX[i], tX, multiply, tX_test, tX_test[i], degree=4)
    tX, tX_test = add_feature(y, tX[i], tX, multiply, tX_test, tX_test[i], degree=5)
    tX, tX_test = add_feature(y, tX[i], tX, multiply, tX_test, tX_test[i], degree=6)
    tX, tX_test = add_feature(y, tX[i], tX, multiply, tX_test, tX_test[i], degree=7)
    tX, tX_test = add_feature(y, tX[i], tX, multiply, tX_test, tX_test[i], degree=8)
    tX, tX_test = add_feature(y, tX[i], tX, multiply, tX_test, tX_test[i], degree=9)
    tX, tX_test = add_feature(y, tX[i], tX, multiply, tX_test, tX_test[i], degree=10)
    tX, tX_test = add_feature(y, tX[i], tX, multiply, tX_test, tX_test[i], degree=11)
    tX, tX_test = add_feature(y, tX[i], tX, multiply, tX_test, tX_test[i], degree=12)
    tX, tX_test = add_feature(y, tX[i], tX, multiply, tX_test, tX_test[i], degree=13)
    tX, tX_test = add_feature(y, tX[i], tX, multiply, tX_test, tX_test[i], degree=14)

tX = tX.T

print("Performing Logistic Regression")
y_binary = np.copy(y)
y_binary[y_binary == -1] = 0
X_log_tr, y_bin_tr, X_log_te, y_bin_te = split_data(tX, y_binary, 0.7)
w_log = logistic_regression_gradient_descent_n(y_bin_tr, X_log_tr, 0.005, 750)

print("Categorizing data between signal and noise")
pred_log = sigmoid(np.dot(tX_test.T, w_log))
pred_log = pred_log[:,0]
pred_log[pred_log >= 0.5] = 1
pred_log[pred_log < 0.5] = -1

print("Writing results to csv")
OUTPUT_PATH = 'csv/submission.csv'
create_csv_submission(ids_test, pred_log, OUTPUT_PATH)
