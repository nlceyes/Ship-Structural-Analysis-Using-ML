# -*- coding: utf-8 -*-
# 读取数据集预测应力结果
import time
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# 超参数
global EPOCH, target_num
EPOCH = 1000
target_num = 5
# FOR SVR: kernel = 'rbf' and epsilon = 0.001 when T1, T2 and T5
models = [	('Linear Regression', LinearRegression()),
			('Support Vector Regression',
			 SVR(kernel = 'linear', C = 100, epsilon = 10)),
			('Ridge Regression', Ridge(alpha = 1)),
			('Nearest Neighbors Regression',
			 KNeighborsRegressor(n_neighbors = 3, weights = 'distance')),
			('Random Forest Regression',
			 RandomForestRegressor(n_estimators = 500)),
			('Gradient Boosting Regression',
			 GradientBoostingRegressor(n_estimators = 500, learning_rate = 0.1, subsample = 0.5)),
			('Extreme Gradient Boosting',
			 xgb.XGBRegressor(n_estimators = 500, learning_rate = 0.1, subsample = 0.5)),
			('LightGBM',
			 lgb.LGBMRegressor(n_estimators = 500, learning_rate = 0.05, subsample = 0.5,
			 num_leaves = 7, min_child_samples = 2))]

def Error_Rate(y_test, y_pred):
	length = len(y_test)
	scores = np.zeros((length, 1))
	for i in range(length):
		scores[i] = abs(y_test[i] - y_pred[i]) / y_test[i]
	return scores.mean()

# R2分数
def Scorer(dataset, targets, model, num):
	scores_r2 = np.zeros((EPOCH, 1))
	scores_mse = np.zeros((EPOCH, 1))
	scores_error = np.zeros((EPOCH, 1))
	for i in range(EPOCH):
		x_train, x_test, y_train, y_test = train_test_split(dataset, targets, test_size = 0.6)
		scaler = StandardScaler().fit(x_train)
		x_train = scaler.transform(x_train)
		x_test = scaler.transform(x_test)
		clf = model[1]
		clf.fit(x_train, y_train)
		y_pred = clf.predict(x_test)
		scores_r2[i] = r2_score(y_test, y_pred)
		scores_mse[i] = mean_squared_error(y_test, y_pred)
		scores_error[i] = Error_Rate(y_test, y_pred)
	print("Target%d, R2: %.2f, MSE: %.2f, \tError: %.2f" % (num+1, 100*scores_r2.mean(), scores_mse.mean(), 100*scores_error.mean()))

# START!
if __name__ == '__main__':
	dd = np.loadtxt('dataset.txt')
	tt = np.loadtxt('target.txt')
	dataset = dd
	for model in models:
		t0 = time.time()
		print('\n%s:' % model[0])
		for i in range(target_num):
			targets = tt[:, i]
			Scorer(dataset, targets, model, i)
		print('All Done in %.3f s' % (time.time() - t0))