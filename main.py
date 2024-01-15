import pandas as pd
import numpy as np
import random

import ast
import sklearn.model_selection

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor

import hackathon_code.Cancer_Breast_Code

if __name__ == '__main__':

    FILE_NAME_TRAIN_FEATURES = 'train.feats.csv'
    FILE_NAME_TRAIN_LABEL0 = 'train.labels.0.csv'
    FILE_NAME_TRAIN_LABEL1 = 'train.labels.1.csv'
    FILE_NAME_TEST_FEATURES = 'test.feats.csv'

    X_train = pd.read_csv(FILE_NAME_TRAIN_FEATURES, low_memory=False)
    X_test = pd.read_csv(FILE_NAME_TEST_FEATURES, low_memory=False)
    y_train_label0 = pd.read_csv(FILE_NAME_TRAIN_LABEL0, low_memory=False)
    y_train_label1 = pd.read_csv(FILE_NAME_TRAIN_LABEL1, low_memory=False)

    Pre_process_train = PreprocessingData(X_train)
    Pre_process_test = PreprocessingData(X_test)

    MLB = MultiLabelBinarizer()

    Model_Multiclassifier = MultiOutputClassifier(DecisionTreeClassifier())

    Model_regression = GradientBoostingRegressor()


    print("Load the data and define the models : Passed 1/8 !")
    #########################################################
    # Part I : predict the location of the distal metastases
    #########################################################
    Pre_process_train.forward_processing()
    X_train = Pre_process_train.X

    Pre_process_test.forward_processing()
    X_test = Pre_process_test.X

    X_test = X_test.reindex(columns = X_train.columns, fill_value = 0)

    print("Pre-process the X-data : Passed 2/8 !")

    y_train_label0 = y_train_label0["אבחנה-Location of distal metastases"].apply(ast.literal_eval).to_numpy()
    y_train_label0_binary = MLB.fit_transform(y_train_label0)

    
    
    print("Pre-process the y0-data : Passed 3/8 !")
    #########################################################

    Model_Multiclassifier.fit(X_train, y_train_label0_binary)

    print("Fit the Multi-label classifier : Passed 4/8 !")

    y_pred_label0_binary = Model_Multiclassifier.predict(X_train)
    y_pred_label0 = pd.DataFrame(MLB.inverse_transform(y_pred_label0_binary))
    y_pred_label0 = y_pred_label0.values.tolist()
    print("Predict y0 : Passed 5/8 !")

    dataframe_pred = pd.DataFrame({'אבחנה-Location of distal metastases': y_pred_label0})
    dataframe_pred['אבחנה-Location of distal metastases'] = [[x for x in inner_list if x is not None] for inner_list in dataframe_pred['אבחנה-Location of distal metastases']]
    pd.DataFrame(dataframe_pred['אבחנה-Location of distal metastases']).to_csv('y_pred_label0.csv', index=False)
    print("Save into a csv file : Passed 6/8 !")
    #########################################################
    #########################################################

    #########################################################
    # Part II : predict the size of the tumor
    #########################################################

    y_train_label1 = y_train_label1['אבחנה-Tumor size'].to_numpy(int)

    print("Pre-process the y1-data : Passed 7/8 !")

    #########################################################

    linear_regression = Model_regression.fit(X_train, y_train_label1)
    y_pred_label1 = np.maximum(0, np.round(linear_regression.predict(X_train), 1))
    pd.DataFrame({'אבחנה-Tumor size': y_pred_label1}).to_csv('y_pred_label1.csv', index=False)
    print("Fit the regression model, predict y1 and save into a csv file : Passed 8/8 !")
    #########################################################
    #########################################################


