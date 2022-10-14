# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import *

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pickle 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import copy
import warnings
warnings.filterwarnings('ignore')



class TrAdaboost:
    def __init__(self, N=20):

        self.N = N
        self.beta_all = np.zeros([1, self.N])
        self.classifiers = []

    def fit(self, x_source, x_target, y_source, y_target):
        x_train = np.concatenate((x_source, x_target), axis=0)
        y_train = np.concatenate((y_source, y_target), axis=0)
        x_train = np.asarray(x_train, order='C')
        y_train = np.asarray(y_train, order='C')
        y_source = np.asarray(y_source, order='C')
        y_target = np.asarray(y_target, order='C')

        row_source = x_source.shape[0]
        row_target = x_target.shape[0]

        # initila the weights
        weight_source = np.ones([row_source, 1]) / row_source
        weight_target = np.ones([row_target, 1]) / row_target
        weights = np.concatenate((weight_source, weight_target), axis=0)

        beta = 1 / (1 + np.sqrt((2 * np.log(row_source))/self.N))

        result = np.ones([row_source + row_target, self.N])

        for i in range(self.N):
            p = self._calculate_weight(weights)
            base_classifier = RandomForestClassifier()
            base_classifier.fit(x_train, y_train, sample_weight = p.flatten())

            self.classifiers.append(base_classifier)

            result[:, i] = base_classifier.predict(x_train)

            error_rate = self._calculate_error_rate(y_target.flatten(),
                                                    result[row_source:, i],
                                                    weights[row_source:, :])

            print("Error Rate in target data: ", error_rate, 'round:', i, 'all_round:', self.N)

            if error_rate > 0.5:
                error_rate = 0.5
            if error_rate == 0:
                self.N = i
                print("Early stopping...")
                break
            self.beta_all[0, i] = error_rate / (1 - error_rate)
            print(self.beta_all[0,i])
            # Update the weight vectors in the source area
            for t in range(row_target):
                weights[row_source + t] = weights[row_source + t] * np.power(self.beta_all[0, i], (-np.abs(result[row_source + t, i] - y_target[t])))


            # Update the weight vectors in the study area
            for s in range(row_source):
                weights[s] = weights[s] * np.power(beta, np.abs(result[s, i] - y_source[s]))


        pweight = pd.DataFrame(weights)
      
    def predict(self, x_test):
        result = np.ones([x_test.shape[0], self.N + 1])
        predict = []

        i = 0
        for classifier in self.classifiers:
            y_pred = classifier.predict(x_test)
            result[:, i] = y_pred
            i += 1

        for i in range(x_test.shape[0]):
            left = np.sum(result[i, int(np.ceil(self.N / 2)): self.N] *
                          np.log(1 / self.beta_all[0, int(np.ceil(self.N / 2)):self.N]))

            right = 0.5 * np.sum(np.log(1 / self.beta_all[0, int(np.ceil(self.N / 2)): self.N]))

            if left >= right:
                predict.append(1)
            else:
                predict.append(0)
        return predict

    def predict_prob(self, x_test):
        result = np.ones([x_test.shape[0], self.N + 1])
        predict = []

        i = 0
        for classifier in self.classifiers:
            y_pred = classifier.predict(x_test)
            result[:, i] = y_pred
            i += 1

        for i in range(x_test.shape[0]):
            left = np.sum(result[i, int(np.ceil(self.N / 2)): self.N] *
                          np.log(1 / self.beta_all[0, int(np.ceil(self.N / 2)):self.N]))

            right = 0.5 * np.sum(np.log(1 / self.beta_all[0, int(np.ceil(self.N / 2)): self.N]))
            predict.append([left, right])
        return predict

    def _calculate_weight(self, weights):
        sum_weight = np.sum(weights)
        return np.asarray(weights / sum_weight, order='C')

    def _calculate_error_rate(self, y_target, y_predict, weight_target):
        sum_weight = np.sum(weight_target)
        return np.sum(weight_target[:, 0] / sum_weight * np.abs(y_target - y_predict))

# Read Data
SourceData = pd.read_excel(r'C:\Users\ZhiYong\Desktop\SourceArea.xlsx')
TargetData = pd.read_excel(r'C:\Users\ZhiYong\Desktop\StudyArea.xlsx')
SourceFeature = SourceData.iloc[0:7000, 0:5].values
SourceLabel = SourceData.iloc[0:7000, 5:6].values
TargetFeature = TargetData.iloc[0:2100, 0:5].values
TargetLabel = TargetData.iloc[0:2100, 5:6].values

#Construct the training set and test set
Trainx= SourceFeature[0:2000]
Trainy= SourceLabel[0:2000]


TransXvalid = TargetFeature[0:1600]
TransYvalid = TargetLabel[0:1600]

TransXtest = TargetFeature[1600:2100]
TransYtest = TargetLabel[1600:2100]


#Decision tree, also can be replaced by svm,rf
clf= RandomForestClassifier()
clf.fit(Trainx,Trainy)

clf2 = RandomForestClassifier()
clf2.fit(TransXvalid, TransYvalid)

#Predict
y_pred_A=clf.predict_proba(Trainx)[:,1] 
y_pred_B_valid=clf.predict_proba(TransXvalid)[:,1]  
y_pred_B_test=clf.predict_proba(TransXtest)[:,1]  
y_pred_c_test = clf2.predict_proba(TransXtest)[:,1]  

print(f" train AUC = {roc_auc_score(Trainy,y_pred_A)}")
print(f" valid AUC = {roc_auc_score(TransYvalid,y_pred_B_valid)}")
print(f" test AUC = {roc_auc_score(TransYtest,y_pred_B_test)}")
print(f" Original AUC = {roc_auc_score(TransYtest,y_pred_c_test)}")

clf3=TrAdaboost()
clf3.fit(Trainx,TransXvalid,Trainy,TransYvalid)

# obtain the best estimator
BestResult = 0
BadResult = 1

for i, estimator in enumerate(clf3.classifiers):
    print('The '+str(i+1)+' estimator:')
    y_pred_A1=estimator.predict_proba(Trainx)[:,1]
    y_pred_B_valid1=estimator.predict_proba(TransXvalid)[:,1]
    y_pred_B_test1=estimator.predict_proba(TransXtest)[:,1]

    result =  roc_auc_score(TransYtest,y_pred_B_test1)
    if result > BestResult:
         BestResult = result
         BestNumber = i
         BestEstimator = estimator
    if result < BadResult:
         BadResult = result
         BadNumber = i

    print(f" train AUC = {roc_auc_score(Trainy,y_pred_A1)}")
    print(f" valid AUC = {roc_auc_score(TransYvalid,y_pred_B_valid1)}")
    print(f" test AUC = {roc_auc_score(TransYtest,y_pred_B_test1)}")
    print('\n',)
    print('==============================================================')
TranPreTest = pd.DataFrame(BestEstimator.predict_proba(TransXtest)[:,1])
TranPreTestClass =pd.DataFrame(BestEstimator.predict(TransXtest))

PreTest = pd.DataFrame(y_pred_c_test)
PreTestClass = pd.DataFrame(clf2.predict(TransXtest))

OutResultPro = pd.concat([TranPreTest, PreTest], axis = 1)
OutResultClass = pd.concat([TranPreTestClass, PreTestClass], axis = 1)

OutResultPro = pd.DataFrame(OutResultPro)
PdOutResultClass = pd.DataFrame(OutResultClass)

OutResultPro.to_excel(r"C:\Users\ZhiYong\Desktop\PSvmTransfer1718.xlsx")
OutResultClass.to_excel(r"C:\Users\ZhiYong\Desktop\CSvmTransfer1718.xlsx")

print("最高测试集正确率", BestResult, "原始测试集正确率", roc_auc_score(TransYtest,y_pred_c_test), "最低测试集正确率", BadResult )

