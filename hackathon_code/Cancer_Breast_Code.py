import pandas as pd
import numpy as np
import random

import ast
import sklearn.model_selection

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
    
class PreprocessingData:
    def __init__(self, X):
        self.X = X
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self.column_names = self.X.columns.tolist()
        
        self.features_numerical = ['אבחנה-Surgery sum', 'אבחנה-Age', 'אבחנה-Nodes exam', 'אבחנה-Positive nodes', 'אבחנה-Tumor depth', 'אבחנה-Tumor width']
        self.features_semi_numerical = ['אבחנה-Her2', 'אבחנה-KI67 protein', 'אבחנה-er', 'אבחנה-pr']
        self.features_datetime = ['אבחנה-Diagnosis date', 'surgery before or after-Activity date']
        self.features_categorical =  ['אבחנה-Basic stage', 'אבחנה-Histological diagnosis', 'אבחנה-Histopatological degree', 'אבחנה-Lymphatic penetration', 'אבחנה-M -metastases mark (TNM)', 'אבחנה-Margin Type', 'אבחנה-N -lymph nodes mark (TNM)', 'אבחנה-Side', 'אבחנה-Stage', 'אבחנה-T -Tumor mark (TNM)', 'surgery before or after-Actual activity', 'אבחנה-Ivi -Lymphovascular invasion']
        self.features_useless = [' Form Name', 'id-hushed_internalpatientid', ' Hospital', 'User Name', 'אבחנה-Surgery date1', 'אבחנה-Surgery name1', 'אבחנה-Surgery date2','אבחנה-Surgery name2', 'אבחנה-Surgery name3', 'אבחנה-Surgery date3']
        self.columns_kept = ['אבחנה-er', 'אבחנה-pr', 'אבחנה-KI67 protein','Date surgery before diagnosis','Date surgery after diagnosis']

    #########################################################
    #########################################################
    def Drop_feature(self, feature_names):
        """
        Drop specified feature(s) from the dataset.
        :param feature_names: Name(s) of the feature(s) to be dropped.
        :return: None
        """
        self.X = self.X.drop(feature_names, axis=1)
    #########################################################
    #########################################################
    def Few_feature(self, feature_names, threshold = 0.7):
        """
         Remove features with high missing/null values from the dataset.
        :param feature_names:  Names of the features to be evaluated.
        :param threshold: Threshold value for the percentage of missing/null values.
            Features exceeding this threshold will be removed. Default is 0.7.
        :return:
        """
        features_high_nb_missing = []
        features_high_nb_null = []
        for column in feature_names:
            if self.X[column].isna().sum() != 0 :
                percent_missing  = self.X[column].isna().sum() / self.n_samples
                if percent_missing > threshold :
                    features_high_nb_missing.append(column)
            else :
                percent_Null = len(self.X[self.X[column] == 'Null']) / self.n_samples
                if percent_Null > threshold :
                    features_high_nb_null.append(column)

        self.Drop_feature(features_high_nb_missing)
        self.Drop_feature(features_high_nb_null)
        
        for i in features_high_nb_missing:
            feature_names.remove(i)

        for i in features_high_nb_null:
            feature_names.remove(i)
            
        return feature_names
    #########################################################
    #########################################################
    def normalize_features(self):
        """
        Normalize the dates of surgeries
        :return: None
        """
        min_age = self.X['Date surgery before diagnosis'].min()
        max_age = self.X['Date surgery before diagnosis'].max()

        normalized_values = []
        for value in self.X['Date surgery before diagnosis']:
            normalized_value = (value - min_age) / (max_age - min_age)
            normalized_values.append(normalized_value)

        self.X['Date surgery before diagnosis'] = normalized_values

        min_age = self.X['Date surgery after diagnosis'].min()
        max_age = self.X['Date surgery after diagnosis'].max()

        normalized_values = []
        for value in self.X['Date surgery after diagnosis']:
            normalized_value = (value - min_age) / (max_age - min_age)
            normalized_values.append(normalized_value)

        self.X['Date surgery after diagnosis'] = normalized_values
    def Date_to_days_feature_datetime(self, feature_names):
        """
        Convert date features to days before and after a reference surgery date.
        :param feature_names: Names of the date features to be converted.
        :return: List of feature names after converting date features to days.
        """
        New_feature_before_days = 'Date surgery before diagnosis'
        New_feature_after_days = 'Date surgery after diagnosis'

        for column in feature_names:
            self.X[column] = pd.to_datetime(self.X[column])

        # Days before
        self.X[New_feature_before_days] = (self.X[feature_names[1]] - self.X[feature_names[0]]).dt.days
        self.X.loc[self.X[New_feature_before_days] < 0, New_feature_before_days] = 0

        # Days After
        self.X[New_feature_after_days] = (self.X[feature_names[1]] - self.X[feature_names[0]]).dt.days
        self.X.loc[self.X[New_feature_after_days] >= 0, New_feature_after_days] = 0
        self.X[New_feature_after_days] = self.X[New_feature_after_days].abs()

        self.Drop_feature(feature_names)
        feature_list = feature_names.copy()
        for i in feature_list:
            feature_names.remove(i)
            
        feature_names.append(New_feature_before_days)
        feature_names.append(New_feature_after_days)

        #######################

        self.normalize_features()
        
        return feature_names
    #########################################################
    #########################################################
    def Mean_fill_feature_numerical(self, feature_names):
        """
        Fill missing values in numerical features with the mean.
        :param feature_names: Names of the numerical features to be filled.
        :return: List of feature names after filling missing values with the mean.
        """
        for column in feature_names:
            self.X[column].fillna(self.X[column].mean(), inplace=True)

            if column == 'אבחנה-Age':
                age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                self.X['אבחנה-Age'] = pd.cut(self.X['אבחנה-Age'], bins=age_bins, labels=labels)
        
        return feature_names
    #########################################################
    #########################################################
    def replace_her2_labels(self):
        """
        Replace HER2 labels in the dataset with binary values.
        :return: None
        """
        positive_terms = ['=', 'pos', 'po', 'yes', '3', '100', 'amplified', 'amp', 'חיובי', '+', 'o']

        negative_terms = ['_', 'no', 'neg', 'non', '1', '2', '+@', 'nef', ',eg', 'meg', 'nrg', 'nec', 'nag', 'nfg', ')', "fish", '0', '-', 'not', "בינוני", "שלילי", "nd"]

        for i, value in enumerate(self.X["אבחנה-Her2"]):
            try:
                lowercase_value = str(value.lower())

                if any(positive in lowercase_value for positive in positive_terms) and \
                        all(negative not in lowercase_value for negative in negative_terms) and \
                        "שלילי" not in lowercase_value and \
                        "1/30%" not in lowercase_value:
                    self.X.at[i, "אבחנה-Her2"] = "1"
                elif any(negative in lowercase_value for negative in negative_terms):
                    self.X.at[i, "אבחנה-Her2"] = "0"
                else:
                    self.X.at[i, "אבחנה-Her2"] = np.NaN

            except:
                self.X.at[i, "אבחנה-Her2"] = np.NaN
    #########################################################
    def replace_er_pr_labels(self, feature):
        """
        Replace ER labels in the dataset with binary values.
        :param feature:
        :return: None
        """
        for i, value in enumerate(self.X[feature]):
            continue_flag = True
            try:
                lowercase_value = str(value.lower())
                for num in range(100, -1, -1):
                    if str(num) in lowercase_value:
                        self.X.at[i, feature] = num / 100
                        continue_flag = False
                        break
                if any(term.lower() in lowercase_value for term in [ "positive", "pos", "חיובי", "+", "high", "3", "pop", "score 4", "strongly po", "po", "strongly" ]) and continue_flag:
                    self.X.at[i, feature] = 1
                elif any(term.lower() in lowercase_value for term in [ "negative", "neg", "שלילי", "-", "_", "low", "netgative", "beg", "nge" ]) and continue_flag:
                    self.X.at[i, feature] = 0
                elif continue_flag :
                    self.X.at[i, feature] = np.NaN
            except:
                self.X.at[i, feature] = np.NaN
    #########################################################
    def replace_KI67_labels(self):
        """
        Replace KI67 labels in the dataset with numeric values.
        :return: None
        """
        for i, value in enumerate(self.X["אבחנה-KI67 protein"]):
            continue_flag = True
            try:
                lowercase_value = str(value.lower())
                for num in range(100, -1, -1):
                    if str(num) in lowercase_value:
                        self.X.at[i, "אבחנה-KI67 protein"] = num / 100
                        continue_flag = False
                        break
                if any(term.lower() in lowercase_value for term in [ "high", "score iv", "interm-high", "pos", "int-h" ]) and continue_flag:
                    self.X.at[i, "אבחנה-KI67 protein"] = 1
                elif any(term.lower() in lowercase_value for term in [ "low", "low-int", "negative", "no" ]) and continue_flag:
                    self.X.at[i, "אבחנה-KI67 protein"] = 0
                elif any(term.lower() in lowercase_value for term in ["score i"]) and continue_flag:
                    self.X.at[i, "אבחנה-KI67 protein"] = 0.25
                elif any(term.lower() in lowercase_value for term in ["score ii","score ii-iii"]) and continue_flag:
                    self.X.at[i, "אבחנה-KI67 protein"] = 0.5
                elif any(term.lower() in lowercase_value for term in ["score iii"]) and continue_flag:
                    self.X.at[i, "אבחנה-KI67 protein"] = 0.75
                elif continue_flag:
                    self.X.at[i, "אבחנה-KI67 protein"] = np.NaN

            except:
                self.X.at[i, "אבחנה-KI67 protein"] = np.NaN
    #########################################################
    def Percent_feature_semi_numerical(self, feature_names):
        """
        Convert percentage-based features to semi-numerical values in the dataset.
        :param feature_names: A list of feature names to be processed.
        :return: The updated list of feature names
        """
        for column in feature_names:
            if column == 'אבחנה-Her2':
                self.replace_her2_labels()

            elif column == 'אבחנה-KI67 protein':
                self.replace_KI67_labels()

            else:
                self.replace_er_pr_labels(column)

        return feature_names
    #########################################################
    #########################################################
    def random_value_column(self, column_name):
        """
        Get a random value from a column in the dataset based on its distribution.
        :param column_name: The name of the column from which to select a random value.
        :return: The randomly selected value from the column.
        """
        value_counts = self.X[column_name].value_counts(normalize=True).to_dict()
        unique_values = list(value_counts.keys())
        probabilities = list(value_counts.values())
        
        random_value = random.choices(unique_values, probabilities)[0]
        
        return random_value
    #########################################################
    def Fill_null_missing_proba(self, feature_names):
        """
        Fill missing values in specified columns with a random value based on their distribution.
        :param feature_names: A list of column names to fill the missing values.
        :return: The list of feature names after filling the missing values.
        """
        for column in feature_names:
            self.X[column] = self.X[column].replace('Null', np.NaN)
            self.X[column].fillna(self.random_value_column(column), inplace=True)
            
        return feature_names
    #########################################################
    #########################################################
    def clean_final_cols(self):
        """
        Clean the columns that will be kept in the final model
        :return: None
        """
        non_integer_columns = self.X.select_dtypes(exclude=['int64, float64']).columns.tolist()
        columns_to_drop = []
        for column in non_integer_columns:
            if column in self.columns_kept:
                continue
            unique_values = self.X[column].nunique()
            if unique_values > 10:
                columns_to_drop.append(column)

        self.X = self.X.drop(columns=columns_to_drop)
        self.features_categorical = [x for x in self.features_categorical if x not in columns_to_drop]

    def forward_processing(self):
            """
            Perform forward processing steps on the dataset.
            :return: None
            """
            self.Drop_feature(self.features_useless)

            self.features_numerical = self.Few_feature(self.features_numerical, threshold=0.8)
            self.features_semi_numerical = self.Few_feature(self.features_semi_numerical, threshold=0.8)
            self.features_datetime = self.Few_feature(self.features_datetime, threshold=0.6)
            self.features_categorical = self.Few_feature(self.features_categorical, threshold=0.6)

            self.features_datetime = self.Date_to_days_feature_datetime(self.features_datetime)
            self.features_numerical = self.Mean_fill_feature_numerical(self.features_numerical)
            self.features_semi_numerical = self.Percent_feature_semi_numerical(self.features_semi_numerical)


            self.features_categorical = self.Fill_null_missing_proba(self.features_categorical)
            self.features_datetime = self.Fill_null_missing_proba(self.features_datetime)
            self.features_semi_numerical = self.Fill_null_missing_proba(self.features_semi_numerical)

            self.clean_final_cols()
            
            self.X.to_csv('filename.csv', index=False)

            self.X = pd.get_dummies(self.X, columns=self.features_categorical)
            self.X = pd.get_dummies(self.X, columns=['אבחנה-Age'])

            
#if __name__ == '__main__':
#
#    FILE_NAME_TRAIN_FEATURES = 'train.feats.csv'
#    FILE_NAME_TRAIN_LABEL0 = 'train.labels.0.csv'
#    FILE_NAME_TRAIN_LABEL1 = 'train.labels.1.csv'
#    FILE_NAME_TEST_FEATURES = 'test.feats.csv'
#
#    X_train = pd.read_csv(FILE_NAME_TRAIN_FEATURES, low_memory=False)
#    X_test = pd.read_csv(FILE_NAME_TEST_FEATURES, low_memory=False)
#    y_train_label0 = pd.read_csv(FILE_NAME_TRAIN_LABEL0, low_memory=False)
#    y_train_label1 = pd.read_csv(FILE_NAME_TRAIN_LABEL1, low_memory=False)
#
#    Pre_process_train = PreprocessingData(X_train)
#    Pre_process_test = PreprocessingData(X_test)
#
#    MLB = MultiLabelBinarizer()
#
#    Model_Multiclassifier = MultiOutputClassifier(DecisionTreeClassifier())
#
#    Model_regression = GradientBoostingRegressor()
#
#
#    print("Load the data and define the models : Passed 1/8 !")
#    #########################################################
#    # Part I : predict the location of the distal metastases
#    #########################################################
#    Pre_process_train.forward_processing()
#    X_train = Pre_process_train.X
#
#    Pre_process_test.forward_processing()
#    X_test = Pre_process_test.X
#
#    X_test = X_test.reindex(columns = X_train.columns, fill_value = 0)
#
#    print("Pre-process the X-data : Passed 2/8 !")
#
#    y_train_label0 = y_train_label0["אבחנה-Location of distal metastases"].apply(ast.literal_eval).to_numpy()
#    y_train_label0_binary = MLB.fit_transform(y_train_label0)
#
#
#
#    print("Pre-process the y0-data : Passed 3/8 !")
#    #########################################################
#
#    Model_Multiclassifier.fit(X_train, y_train_label0_binary)
#
#    print("Fit the Multi-label classifier : Passed 4/8 !")
#
#    y_pred_label0_binary = Model_Multiclassifier.predict(X_train)
#    y_pred_label0 = pd.DataFrame(MLB.inverse_transform(y_pred_label0_binary))
#    y_pred_label0 = y_pred_label0.values.tolist()
#    print("Predict y0 : Passed 5/8 !")
#
#    dataframe_pred = pd.DataFrame({'אבחנה-Location of distal metastases': y_pred_label0})
#    dataframe_pred['אבחנה-Location of distal metastases'] = [[x for x in inner_list if x is not None] for inner_list in dataframe_pred['אבחנה-Location of distal metastases']]
#    pd.DataFrame(dataframe_pred['אבחנה-Location of distal metastases']).to_csv('y_pred_label0.csv', index=False)
#    print("Save into a csv file : Passed 6/8 !")
#    #########################################################
#    #########################################################
#
#    #########################################################
#    # Part II : predict the size of the tumor
#    #########################################################
#
#    y_train_label1 = y_train_label1['אבחנה-Tumor size'].to_numpy(int)
#
#    print("Pre-process the y1-data : Passed 7/8 !")
#
#    #########################################################
#
#    linear_regression = Model_regression.fit(X_train, y_train_label1)
#    y_pred_label1 = np.maximum(0, np.round(linear_regression.predict(X_train), 1))
#    pd.DataFrame({'אבחנה-Tumor size': y_pred_label1}).to_csv('y_pred_label1.csv', index=False)
#    print("Fit the regression model, predict y1 and save into a csv file : Passed 8/8 !")
#    #########################################################
#    #########################################################





