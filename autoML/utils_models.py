import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, \
    AdaBoostRegressor, GradientBoostingRegressor, GradientBoostingClassifier, \
    ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import RANSACRegressor, LinearRegression, Ridge, Lasso, ElasticNet, \
    LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, SGDRegressor, \
    PassiveAggressiveRegressor, LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron, \
    PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from hyperopt import hp


# 17
Classifiers_model_map = {'LogisticRegression': LogisticRegression,
                         'RandomForestClassifier': RandomForestClassifier,
                         'RidgeClassifier': RidgeClassifier,
                         'GradientBoostingClassifier': GradientBoostingClassifier,
                         'ExtraTreesClassifier': ExtraTreesClassifier,
                         'AdaBoostClassifier': AdaBoostClassifier,
                         'BaggingClassifier': BaggingClassifier,
                         'SVC': SVC,
                         'SGDClassifier': SGDClassifier,
                         'PassiveAggressiveClassifier': PassiveAggressiveClassifier,
                         'KNN': KNeighborsClassifier,
                         'GaussianProcessClassifier': GaussianProcessClassifier,
                         'GaussianNB': GaussianNB,
                         'DecisionTreeClassifier': tree.DecisionTreeClassifier,
                         'Perceptron': Perceptron,
                         'XGBClassifier': XGBClassifier,
                         'LGBMClassifier': LGBMClassifier,
                         }

# 19
Regressors_model_map = {'LinearRegression': LinearRegression,
                        'RandomForestRegressor': RandomForestRegressor,
                        'Ridge': Ridge,
                        'SVR': SVR,
                        'ExtraTreesRegressor': ExtraTreesRegressor,
                        'AdaBoostRegressor': AdaBoostRegressor,
                        'RANSACRegressor': RANSACRegressor,
                        'GradientBoostingRegressor': GradientBoostingRegressor,
                        'Lasso': Lasso,
                        'ElasticNet': ElasticNet,
                        'LassoLars': LassoLars,
                        'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit,
                        'BayesianRidge': BayesianRidge,
                        'ARDRegression': ARDRegression,
                        'PassiveAggressiveRegressor': PassiveAggressiveRegressor,
                        'GaussianProcessRegressor': GaussianProcessRegressor,
                        'XGBRegressor': XGBRegressor,
                        'LGBMRegressor': LGBMRegressor,
                        'SGDRegressor': SGDRegressor,
                        }

# test_classifiers = ['KNN', 'LogisticRegression', 'GaussianNB', 'RandomForestClassifier',
#                     'DecisionTreeClassifier', 'GradientBoostingClassifier', 'ExtraTreesClassifier',
#                     'AdaBoostClassifier', 'Perceptron', 'BaggingClassifier', 'SGDClassifier', 'XGBClassifier',
#                     'LGBMClassifier', 'SVC', 'PassiveAggressiveClassifier',
#                     'RidgeClassifier', 'GaussianProcessClassifier']  # 17
test_classifiers = ['KNN', 'LogisticRegression', 'GaussianNB', 'RandomForestClassifier',
                    'DecisionTreeClassifier', 'GradientBoostingClassifier', 'ExtraTreesClassifier',
                    'AdaBoostClassifier', 'Perceptron', 'BaggingClassifier', 'SGDClassifier', 'XGBClassifier',
                    'LGBMClassifier', 'SVC', 'PassiveAggressiveClassifier',
                    'RidgeClassifier', 'GaussianProcessClassifier']


# test_Regressions = ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'SVR',
#                     'AdaBoostRegressor', 'RANSACRegressor', 'GradientBoostingRegressor', 'Lasso',
#                     'ElasticNet', 'LassoLars', 'OrthogonalMatchingPursuit', 'BayesianRidge',
#                     'PassiveAggressiveRegressor', 'XGBRegressor', 'LGBMRegressor',
#                     'SGDRegressor', 'ARDRegression', 'Ridge', 'GaussianProcessRegressor']  # 19
test_Regressions = ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'SVR',
                    'AdaBoostRegressor', 'RANSACRegressor', 'GradientBoostingRegressor', 'Lasso',
                    'ElasticNet', 'LassoLars', 'OrthogonalMatchingPursuit', 'BayesianRidge',
                    'PassiveAggressiveRegressor', 'XGBRegressor', 'LGBMRegressor',
                    'SGDRegressor', 'ARDRegression', 'Ridge', 'GaussianProcessRegressor']




search_params = {
    'KNN': {
        'n_neighbors': {'search': 'choice', 'space': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 20, 50]},
        'weights': {'search': 'choice', 'space': ['uniform', 'distance']},
        'algorithm': {'search': 'choice', 'space': ['auto', 'ball_tree', 'kd_tree', 'brute']},
        'p': {'search': 'choice', 'space': [1, 2, 3, 4, 5, 6]},
        # 'n_jobs': {'search': 'choice', 'space': [-1]}
    },
    'LogisticRegression': {
        'solver': {'search': 'choice', 'space': ['newton-cg', 'lbfgs', 'sag', 'saga']},
        'C': {'search': 'choice', 'space': [.0001, .001, .01, .1, 1, 10, 100, 1000]},
        'multi_class': {'search': 'choice', 'space': ['ovr', 'multinomial', 'auto']},
        'class_weight': {'search': 'choice', 'space': [None, 'balanced']},
        'fit_intercept': {'search': 'choice', 'space': [True, False]},
        'max_iter': {'search': 'choice', 'space': [5000]}
    },
    'GaussianNB': {
        'priors': {'search': 'choice', 'space': [None]}
    },
    'RandomForestClassifier': {
        'n_estimators': {'search': 'choice', 'space': [20, 50, 90, 100, 110, 200, 500]},
        'criterion': {'search': 'choice', 'space': ['entropy', 'gini']},
        'class_weight': {'search': 'choice', 'space': [None, 'balanced']},
        'max_features': {'search': 'choice', 'space': ['auto', 'sqrt', 'log2', None]},
        'min_samples_split': {'search': 'choice', 'space': [2, 5, 20, 50, 100]},
        'min_samples_leaf': {'search': 'choice', 'space': [1, 2, 5, 20, 50, 100]},
        'bootstrap': {'search': 'choice', 'space': [True, False]},
    },
    'DecisionTreeClassifier': {
        'criterion': {'search': 'choice', 'space': ['entropy', 'gini']},
        'splitter': {'search': 'choice', 'space': ['best', 'random']},
        'max_features': {'search': 'choice', 'space': ['sqrt', 'log2', None]},
    },
    'GradientBoostingClassifier': {
        # 'loss': {'search': 'choice', 'space': ['deviance', 'exponential']},
        'max_depth': {'search': 'choice', 'space': [1, 2, 3, 4, 5, 7, 10, 15]},
        'max_features': {'search': 'choice', 'space': ['sqrt', 'log2', None]},
        'learning_rate': {'search': 'uniform', 'space': [0.001, 0.2]},
        'subsample': {'search': 'uniform', 'space': [0.5, 1.0]},
        'n_estimators': {'search': 'choice', 'space': [10, 50, 75, 100, 125, 150, 200, 500]},
    },
    'ExtraTreesClassifier': {
        'n_estimators': {'search': 'choice', 'space': [10, 12, 14, 15, 20, 50, 100, 200, 500]}
    },
    'AdaBoostClassifier': {
        'n_estimators': {'search': 'choice', 'space': [40, 50, 55, 60]},
        'learning_rate': {'search': 'uniform', 'space': [0.7, 0.8, 0.85, 0.9, 0.95, 1]}
    },
    'Perceptron': {
        'penalty': {'search': 'choice', 'space': ['none', 'l2', 'l1', 'elasticnet']},
        'alpha': {'search': 'uniform', 'space': [.0000001, .000001, .00001, .0001, .001]},
        'class_weight': {'search': 'choice', 'space': ['balanced', None]}
    },
    'BaggingClassifier': {
        'n_estimators': {'search': 'choice', 'space': [10, 12, 14, 15, 20, 50, 100, 200, 500]},
        'max_samples': {'search': 'uniform', 'space': [0.5, 1]},
        'max_features': {'search': 'uniform', 'space': [0.5, 1]},
        'bootstrap': {'search': 'choice', 'space': [True, False]},
    },
    'SGDClassifier': {
        'loss': {'search': 'choice', 'space': [
            'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss',
            'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']},
        'penalty': {'search': 'choice', 'space': ['none', 'l2', 'l1', 'elasticnet']},
        'alpha': {'search': 'uniform', 'space': [.0000001, .000001, .00001, .0001, .001]},
        'learning_rate': {'search': 'choice', 'space': ['constant', 'optimal', 'invscaling']},
        'eta0': {'search': 'uniform', 'space': [0.0001, 0.2]},
        'class_weight': {'search': 'choice', 'space': ['balanced', None]},
        'max_iter': {'search': 'choice', 'space': [5000]}
    },
    'XGBClassifier': {
        'objective': {'search': 'choice', 'space': ['reg:squarederror']},
        'base_score': {'search': 'uniform', 'space': [0.4, 0.6]},
        'max_depth': {'search': 'choice', 'space': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]},
        'learning_rate': {'search': 'uniform', 'space': [0.001, 0.2]},
        'booster': {'search': 'choice', 'space': ['gbtree', 'gblinear', 'dart']},
        'n_estimators': {'search': 'choice', 'space': [50, 75, 100, 150, 200, 375, 500]},
        'min_child_weight': {'search': 'choice', 'space': [1, 2, 3, 4, 5, 6, 7, 10]},
        'subsample': {'search': 'uniform', 'space': [0.5, 1.0]},
        'colsample_bytree': {'search': 'uniform', 'space': [0.5, 0.8, 1.0]}
        # 'lambda': [0.9, 1.0]

    },
    'LGBMClassifier': {
        'boosting_type': {'search': 'choice', 'space': ['gbdt', 'dart']},
        # 'min_child_samples': {'search': 'choice', 'space': [1, 5, 7, 10, 15, 20, 35, 50]},
        'num_leaves':
            {'search': 'choice', 'space':
                [2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250]},
        'colsample_bytree': {'search': 'uniform', 'space': [0.7, 0.9, 1.0]},
        'subsample': {'search': 'uniform', 'space': [0.7, 0.9, 1.0]},
        'learning_rate': {'search': 'uniform', 'space': [0.001, 0.2]},
        'n_estimators': {'search': 'choice', 'space': [50, 75, 100, 150, 200, 350, 500]},
        # 'min_data': {'search': 'choice', 'space': [1]},
        # 'min_data_in_bin': {'search': 'choice', 'space': [1]}
    },
    'SVC': {
        'kernel': {'search': 'choice', 'space': ['linear', 'poly', 'rbf', 'sigmoid']},
        'degree': {'search': 'choice', 'space': [1, 2, 3, 4, 5, 6, 7]},
        'C': {'search': 'uniform', 'space': [0.1, 5]},
        'gamma': {'search': 'choice', 'space': ['auto', 'scale']}
    },
    'PassiveAggressiveClassifier': {
        'loss': {'search': 'choice', 'space': ['hinge', 'squared_hinge']},
        'class_weight': {'search': 'choice', 'space': ['balanced', None]},
        'C': {'search': 'uniform', 'space': [0.01, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]}
    },
    'RidgeClassifier': {
        'alpha': {'search': 'uniform', 'space': [.0001, .001, .01, .1, 1, 10, 100, 1000]},
        'class_weight': {'search': 'choice', 'space': [None, 'balanced']},
        'solver': {'search': 'choice', 'space': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']}
    },
    'GaussianProcessClassifier': {
        'kernel': {'search': 'choice', 'space': [None]}
    },
    'LinearRegression': {
        'fit_intercept': {'search': 'choice', 'space': [True, False]},
        'normalize': {'search': 'choice', 'space': [True, False]}
    },
    # regressor algorithm
    'RandomForestRegressor': {
        'n_estimators': {'search': 'choice', 'space': [20, 50, 90, 100, 110, 200, 500]},
        'max_features': {'search': 'choice', 'space': ['auto', 'sqrt', 'log2', None]},
        'min_samples_split': {'search': 'choice', 'space': [2, 5, 20, 50, 100]},
        'min_samples_leaf': {'search': 'choice', 'space': [1, 2, 5, 20, 50, 100]},
        'bootstrap': {'search': 'choice', 'space': [True, False]}
    },
    'ExtraTreesRegressor': {
        'n_estimators': {'search': 'choice', 'space': [20, 50, 90, 100, 110, 200, 500]},
        'max_features': {'search': 'choice', 'space': ['auto', 'sqrt', 'log2', None]},
        'min_samples_split': {'search': 'choice', 'space': [2, 5, 20, 50, 100]},
        'min_samples_leaf': {'search': 'choice', 'space': [1, 2, 5, 20, 50, 100]},
        'bootstrap': {'search': 'choice', 'space': [True, False]}
    },
    'SVR': {
        'epsilon': {'search': 'uniform', 'space': [0, 0.2]},
        'kernel': {'search': 'choice', 'space': ['linear', 'poly', 'rbf', 'sigmoid']},
        'degree': {'search': 'choice', 'space': [1, 2, 3, 4, 5, 6, 7]},
        'C': {'search': 'uniform', 'space': [0.1, 5]},
        'gamma': {'search': 'choice', 'space': ['auto', 'scale']}
    },
    'AdaBoostRegressor': {
        'loss': {'search': 'choice', 'space': ['linear', 'square', 'exponential']},
        'n_estimators': {'search': 'choice', 'space': [30, 40, 45, 50, 55, 60, 70, 90, 100]},
        'learning_rate': {'search': 'uniform', 'space': [0.7, 0.8, 0.85, 0.9, 0.95, 1, 1.5]}
    },
    'RANSACRegressor': {
        'min_samples': {'search': 'uniform', 'space': [0.5, 1]},
        'stop_probability': {'search': 'uniform', 'space': [0.99, 0.98, 0.95, 0.90]}
    },
    'GradientBoostingRegressor': {
        # Add in max_delta_step if classes are extremely imbalanced
        'max_depth': {'search': 'choice', 'space': [1, 2, 3, 4, 5, 7, 10, 15]},
        'max_features': {'search': 'choice', 'space': ['sqrt', 'log2', None]},
        'loss': {'search': 'choice', 'space': ['ls', 'lad', 'huber', 'quantile']},
        'learning_rate': {'search': 'uniform', 'space': [0.001, 0.2]},
        'n_estimators': {'search': 'choice', 'space': [20, 50, 90, 100, 110, 200, 500]},
        'subsample': {'search': 'uniform', 'space': [0.5, 1.0]},
    },
    'Lasso': {
        'selection': {'search': 'choice', 'space': ['cyclic', 'random']},
        'tol': {'search': 'uniform', 'space': [.0000001, .000001, .00001, .0001, .001]},
        'positive': {'search': 'choice', 'space': [True, False]},
        'max_iter': {'search': 'choice', 'space': [50, 100, 250, 500, 1000]}
    },
    'ElasticNet': {
        'l1_ratio': {'search': 'uniform', 'space': [0.1, 0.3, 0.5, 0.7, 0.9]},
        'selection': {'search': 'choice', 'space': ['cyclic', 'random']},
        'tol': {'search': 'uniform', 'space': [.0000001, .000001, .00001, .0001, .001]},
        'positive': {'search': 'choice', 'space': [True, False]}
    },
    'LassoLars': {
        'positive': {'search': 'choice', 'space': [True, False]},
        'max_iter': {'search': 'choice', 'space': [1000]}
    },
    'OrthogonalMatchingPursuit': {
        'n_nonzero_coefs': {'search': 'choice', 'space': [None, 3, 5, 10, 25]},
        'fit_intercept': {'search': 'choice', 'space': [True, False]},
        'normalize': {'search': 'choice', 'space': [True, False]},
        'precompute': {'search': 'choice', 'space': [True, False, 'auto']}
    },
    'BayesianRidge': {
        'tol': {'search': 'uniform', 'space': [.000001, .00001, .0001, .001, 0.01]},
        'alpha_1': {'search': 'uniform', 'space': [.0000001, .000001, .00001, .0001, .001]},
        'lambda_1': {'search': 'uniform', 'space': [.0000001, .000001, .00001, .0001, .001]},
        'lambda_2': {'search': 'uniform', 'space': [.0000001, .000001, .00001, .0001, .001]},
        'fit_intercept': {'search': 'choice', 'space': [True, False]},
        'normalize': {'search': 'choice', 'space': [True, False]}
    },
    'PassiveAggressiveRegressor': {
        'tol': {'search': 'uniform', 'space': [.000001, .00001, .0001, .001, 0.01]},
        'epsilon': {'search': 'uniform', 'space': [0.01, 0.05, 0.1, 0.2, 0.5]},
        'loss': {'search': 'choice', 'space': ['epsilon_insensitive', 'squared_epsilon_insensitive']},
        # 'C': {'search': 'choice', 'space': [.0001, .001, .01, .1, 1]},
    },
    'XGBRegressor': {
        # Add in max_delta_step if classes are extremely imbalanced
        'base_score': {'search': 'uniform', 'space': [0.4, 0.6]},
        'max_depth': {'search': 'choice', 'space': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]},
        'learning_rate': {'search': 'uniform', 'space': [0.001, 0.2]},
        'booster': {'search': 'choice', 'space': ['gbtree', 'gblinear', 'dart']},
        'n_estimators': {'search': 'choice', 'space': [50, 75, 100, 150, 200, 375, 500]},
        'min_child_weight': {'search': 'choice', 'space': [1, 2, 3, 4, 5, 6, 7, 10]},
        'subsample': {'search': 'uniform', 'space': [0.5, 1.0]},
        'objective': {'search': 'choice', 'space': ['reg:squarederror']},
        'colsample_bytree': {'search': 'uniform', 'space': [0.5, 0.8, 1.0]}
    },
    'LGBMRegressor': {
        'boosting_type': {'search': 'choice', 'space': ['gbdt', 'dart']},
        # 'min_child_samples': {'search': 'choice', 'space': [1, 5, 7, 10, 15, 20, 35, 50]},
        'num_leaves':
            {'search': 'choice', 'space':
                [10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250]},
        'colsample_bytree': {'search': 'uniform', 'space': [0.7, 0.9, 1.0]},
        'subsample': {'search': 'uniform', 'space': [0.7, 0.9, 1.0]},
        'learning_rate': {'search': 'uniform', 'space': [0.001, 0.2]},
        'n_estimators': {'search': 'choice', 'space': [50, 75, 100, 150, 200, 350, 500]},
        # 'min_data': {'search': 'choice', 'space': [1]},
        # 'min_data_in_bin': {'search': 'choice', 'space': [1]},

    },
    'SGDRegressor': {
        'loss': {'search': 'choice', 'space':
            ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']},
        'penalty': {'search': 'choice', 'space':
            ['none', 'l2', 'l1', 'elasticnet']},
        'learning_rate': {'search': 'choice', 'space': ['constant', 'optimal', 'invscaling', 'adaptive']},
        'eta0': {'search': 'uniform', 'space': [0.0001, 0.2]},
        'alpha': {'search': 'uniform', 'space': [.0000001, .000001, .00001, .0001, .001]},
        'max_iter': {'search': 'choice', 'space': [5000]}
    },
    'ARDRegression': {
        'n_iter': {'search': 'choice', 'space': [100, 200, 300, 400, 500]},
        'tol': {'search': 'uniform', 'space': [.0000001, .000001, .00001, .0001, .001]},
        'alpha_1': {'search': 'uniform', 'space': [.0000001, .000001, .00001, .0001, .001]},
        'alpha_2': {'search': 'uniform', 'space': [.0000001, .000001, .00001, .0001, .001]},
        'lambda_1': {'search': 'uniform', 'space': [.0000001, .000001, .00001, .0001, .001]},
        'lambda_2': {'search': 'uniform', 'space': [.0000001, .000001, .00001, .0001, .001]},
        'threshold_lambda': {'search': 'choice', 'space': [100, 1000, 10000, 100000, 1000000]}
    },
    'Ridge': {
        'alpha': {'search': 'uniform', 'space': [.0001, .001, .01, .1, 1, 10, 100, 1000]},
        'solver': {'search': 'choice', 'space': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']}
    },
    'GaussianProcessRegressor': {
        'kernel': {'search': 'choice', 'space': [None]}
    },
}


def run_classifiers(X_df, y, cv_split):
    results_columns = ['model_name', 'train_accuracy', 'test_accuracy', 'train_time', 'test_time']
    results_df = pd.DataFrame(columns=results_columns)
    row_index = 0
    for classifier in test_classifiers:
        results_df.loc[row_index, 'model_name'] = classifier
        print('****** %s ******' % classifier)
        model = Classifiers_model_map[classifier]()
        scores = cross_validate(model, X_df, y, cv=cv_split, scoring='accuracy', return_train_score=True, n_jobs=-1)
        train_time = round(np.mean(scores['fit_time']), 4)
        test_time = round(np.mean(scores['score_time']), 4)
        train_accuracy = round(np.mean(scores['train_score']), 4)
        test_accuracy = round(np.mean(scores['test_score']), 4)
        results_df.loc[row_index, 'train_accuracy'] = train_accuracy
        results_df.loc[row_index, 'test_accuracy'] = test_accuracy
        results_df.loc[row_index, 'train_time'] = train_time
        results_df.loc[row_index, 'test_time'] = test_time
        row_index += 1
    # 依据模型的准确率进行排序，找到所有模型中表现最好的那个模型
    results_df.sort_values(by=['test_accuracy'], ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    results_df.to_csv('results.csv', encoding='ANSI', index=None)
    # best_model = Classifiers_model_map[results_df['model_name'].loc[0]]
    # search_space = search_params[results_df['model_name'].loc[0]]
    # 2019-8-20修改为返回最好的三个模型
    candidate_model_list = [Classifiers_model_map[results_df['model_name'].loc[i]] for i in range(3)]
    candidate_search_space_list = [search_params[results_df['model_name'].loc[i]] for i in range(3)]
    return candidate_model_list, candidate_search_space_list


def run_regressions(X_df, y, cv_split):
    results_columns = ['model_name', 'train_r2', 'test_r2', 'train_MSE', 'test_MSE',
                       'train_MAE', 'test_MAE', 'train_time', 'test_time']
    results_df = pd.DataFrame(columns=results_columns)
    scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    row_index = 0
    for regres in test_Regressions:
        print('****** %s ******' % regres)
        results_df.loc[row_index, 'model_name'] = regres
        model = Regressors_model_map[regres]()
        scores = cross_validate(model, X_df, y, scoring=scoring,
                                cv=cv_split, return_train_score=True)
        train_time = round(np.mean(scores['fit_time']), 4)
        test_time = round(np.mean(scores['score_time']), 4)
        train_r2 = round(np.mean(scores['train_r2']), 4)
        test_r2 = round(np.mean(scores['test_r2']), 4)
        train_MSE = abs(round(np.mean(scores['train_neg_mean_squared_error']), 4))
        test_MSE = abs(round(np.mean(scores['test_neg_mean_squared_error']), 4))
        train_MAE = abs(round(np.mean(scores['train_neg_mean_absolute_error']), 4))
        test_MAE = abs(round(np.mean(scores['test_neg_mean_absolute_error']), 4))

        results_df.loc[row_index, 'train_r2'] = train_r2
        results_df.loc[row_index, 'test_r2'] = test_r2
        results_df.loc[row_index, 'train_time'] = train_time
        results_df.loc[row_index, 'test_time'] = test_time
        results_df.loc[row_index, 'train_MSE'] = train_MSE
        results_df.loc[row_index, 'test_MSE'] = test_MSE
        results_df.loc[row_index, 'train_MAE'] = train_MAE
        results_df.loc[row_index, 'test_MAE'] = test_MAE
        row_index += 1
    # 依据模型的R2进行排序，找到所有模型中表现最好的那个模型
    results_df.sort_values(by=['test_r2'], ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    results_df.to_csv('results.csv', encoding='UTF-8', index=None)

    # best_model = Regressors_model_map[results_df['model_name'].loc[0]]
    # search_space = search_params[results_df['model_name'].loc[0]]
    # 2019-8-20修改为返回最好的三个模型
    candidate_model_list = [Regressors_model_map[results_df['model_name'].loc[i]] for i in range(3)]
    candidate_search_space_list = [search_params[results_df['model_name'].loc[i]] for i in range(3)]
    return candidate_model_list, candidate_search_space_list
