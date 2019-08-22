from autoML.predictor import Predictor
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, make_classification
import warnings
warnings.filterwarnings(action='ignore')

# df_train = pd.read_csv("first_senors.csv", encoding="ANSI")
# df_test = pd.read_csv("first_senors-test.csv", encoding="ANSI")
# iris = datasets.load_iris()
# X = iris['data']
# y = iris['target']
# 分类任务
X, y = make_classification(n_samples=1000, n_features=50, n_clusters_per_class=1, n_classes=4)
train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=0.8, random_state=None)
df_train = pd.concat([pd.DataFrame(train_x), pd.Series(train_y)], axis=1, ignore_index=True)
df_test = pd.DataFrame(test_x)
# 指定要预测的列为output
column_descriptions = {df_train.columns[-1]: 'output'}
ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
# 指定任务，为分类'classifier 还是回归'regressor'
ml_predictor.train(df_train, cv_n=5, feature_selection=True, pca=True)
# 预测
pre = ml_predictor.predict(pd.DataFrame(test_x))
print("分类模型测试集的准确率为：{}".format(sum(pre == test_y) / len(pre)))

# 回归任务
# X, y = make_regression(n_samples=1000, n_features=50, random_state=None)
# train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=0.8, random_state=66)
# df_train = pd.concat([pd.DataFrame(train_x), pd.Series(train_y)], axis=1, ignore_index=True)
# df_test = pd.DataFrame(test_x)
# column_descriptions = {df_train.columns[-1]: 'output'}
# # 指定任务，为分类'classifier 还是回归'regressor'
# ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)
# ml_predictor.train(df_train, cv_n=5, feature_selection=True, pca=True)
# pre = ml_predictor.predict(pd.DataFrame(test_x))
# print("回归模型测试集的RMSE为：{}".format((sum((pre - test_y) ** 2) / len(pre)) ** 0.5))
# print("回归模型测试集的MAE为：{}".format(sum(np.absolute(pre - test_y)) / len(pre)))

