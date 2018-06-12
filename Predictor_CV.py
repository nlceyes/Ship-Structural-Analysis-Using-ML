# -*- coding: utf-8 -*-
# 读取数据集，调参并返回最佳参数与R2_Score
import time
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from collections import Counter

# 超参数
global EPOCH, target_num
EPOCH = 100
target_num = 5
model_list = [('Support Vector Regression', SVR(),
				{'kernel': ['rbf', 'linear'],
				'C': [100, 10, 1, 0.1, 0.01],
				'epsilon': [10, 1, 1e-1, 1e-2, 1e-3]}
				),
				('Ridge Regression', Ridge(),
				{'alpha': [100, 10, 1, 0.1, 0.01]}
				),
				('Lasso Regression', Lasso(),
				{'alpha': [100, 10, 1, 0.1, 0.01]}
				),
				('Nearest Neighbors Regression', KNeighborsRegressor(),
				{'n_neighbors': [1, 3, 5, 7, 9],
				'weights': ['uniform', 'distance']}
				),
				('Random Forest Regression', RandomForestRegressor(),
				{'n_estimators': [10, 50, 100, 200, 500],
				'max_features': ['auto', 'sqrt', 'log2']}
				),
				('Gradient Boosting Regression', GradientBoostingRegressor(),
				{'n_estimators': [10, 50, 100, 200, 500],
				'learning_rate': [0.2, 0.1, 0.05, 0.01],
				'subsample': [0.5, 0.75, 1.0],
				'max_features': ['auto', 'sqrt', 'log2']}
				),
				('Extreme Gradient Boosting', xgb.XGBRegressor(),
				{'n_estimators': [10, 50, 100, 200, 500],
				'learning_rate': [0.2, 0.1, 0.05, 0.01],
				'max_depth': [1, 3, 5],
				'gamma': [0, 0.1, 0.3],
				'subsample': [0.5, 0.75, 1.0],
				'booster': ['gbtree', 'gblinear']}
				),
				('LightGBM', lgb.LGBMRegressor(),
				{'num_leaves': [7, 15, 31],
				'learning_rate': [0.2, 0.1, 0.05, 0.01],
				'n_estimators': [10, 50, 100, 200, 500],
				'min_split_gain': [0, 0.1, 0.3],
				'min_child_samples': [2, 5, 10],
				'subsample': [0.5, 0.75, 1.0]}
				)]

# 调参并预测
def tuning_and_scoring(dataset, targets, model, param_dict_, param_key):
	t0 = time.time()
	scores = np.zeros((EPOCH, 1))
	for i in range(EPOCH):
		x_train, x_test, y_train, y_test = train_test_split(dataset, targets, test_size = 0.6)
		scaler = StandardScaler().fit(x_train)
		x_train = scaler.transform(x_train)
		x_test = scaler.transform(x_test)
		clf = GridSearchCV(model[1], model[2], cv=4, scoring='r2', n_jobs = 4)
		clf.fit(x_train, y_train)
		temp = clf.best_params_
		print('%d:\t%s' % (i+1, temp))
		for param in param_key:
			param_dict_[param].append(temp[param])
		y_pred = clf.predict(x_test)
		scores[i] = r2_score(y_test, y_pred)
	print("Scores: %.3f" % scores.mean())
	print('All Done in %.3f s' % (time.time() - t0))
	return scores.mean(), param_dict_

if __name__ == '__main__':
	param_dict = {}
	score_dict = {}
	# 读取数据集
	dd = np.loadtxt('dataset.txt')
	tt = np.loadtxt('target.txt')
	dataset = dd
	for model in model_list:
		print('\n%s:' % model[0])
		param_dict[model[0]] = {}
		score_dict[model[0]] = {}
		for i in range(target_num):
			print('Target%d:' % (i+1))
			param_dict[model[0]][str(i+1)] = {}
			param_key = list(model[2])
			for param in param_key:
				param_dict[model[0]][str(i+1)][param] = []
			targets = tt[:, i]
			score_dict[model[0]][str(i+1)], param_dict[model[0]][str(i+1)] = tuning_and_scoring(dataset, targets, model, param_dict[model[0]][str(i+1)], param_key)
	with open('result.txt', 'w') as f:
		f.write('%s\n\n%s' % (score_dict, param_dict))
	# 获取出现最多的超参数
	index_models = list(param_dict)
	index_targets = ['1', '2', '3', '4', '5']
	for index_model in index_models:
		print('\n%s:' % index_model)
		results_all = param_dict[index_model]
		for index_target in index_targets:
			results_target = results_all[index_target]
			index_params = list(results_target)
			for index_param in index_params:
				results_param = results_target[index_param]
				counter_param = Counter(results_param).most_common(3)
				print('Target%s%s%s' % (index_target.ljust(5), index_param.ljust(15), counter_param))