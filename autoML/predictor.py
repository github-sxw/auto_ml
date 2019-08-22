import pandas as pd
import time
import os
import warnings
import shutil
import numpy as np
import joblib
from hyperopt import fmin, tpe, hp, partial, STATUS_OK
from autoML import utils_models
from autoML import utils_data_cleaning
from autoML import utils_feature_selection
from sklearn.model_selection import cross_validate, ShuffleSplit, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


class Predictor(object):
    def __init__(self, type_of_estimator, column_descriptions):
        if type_of_estimator.lower() in [
            'regressor', 'regression', 'regressions', 'regressors', 'number', 'numeric',
            'continuous'
        ]:
            self.type_of_estimator = 'regressor'
        elif type_of_estimator.lower() in [
            'classifier', 'classification', 'categorizer', 'categorization', 'categories',
            'labels', 'labeled', 'label'
        ]:
            self.type_of_estimator = 'classifier'
        else:
            print('Invalid value for "type_of_estimator". Please pass in either "regressor" or '
                  '"classifier". You passed in: ' + type_of_estimator)
            raise ValueError(
                'Invalid value for "type_of_estimator". Please pass in either "regressor" or '
                '"classifier". You passed in: ' + type_of_estimator)
        self.column_descriptions = column_descriptions
        self.best_model = None
        self.best_model_space = None
        # self.select_index = None
        self._validate_input_col_descriptions()
        self.feature_selection = False
        self.scaler = True
        self.pca = True

    # 验证是否输入了需要预测的列名
    def _validate_input_col_descriptions(self):
        found_output_column = False
        for key, value in self.column_descriptions.items():
            value = value.lower()
            if value == 'output':
                self.output_column = key
                found_output_column = True
        if found_output_column is False:
            print('Here is the column_descriptions that was passed in:')
            print(self.column_descriptions)
            raise ValueError(
                'In your column_descriptions, please make sure exactly one column has the value '
                '"output", which is the value we will be training models to predict. ')

    def get_X_and_y(self, data):
        # 如果数据是列表形式，就转换为DataFrame
        if isinstance(data, list):
            X_df = pd.DataFrame(data)
            del data
        else:
            X_df = data.copy()
        # 分离X和y
        y = X_df[self.output_column]
        X_df.drop(self.output_column, axis=1, inplace=True)

        return X_df, y

    def _clean_data(self, X, y=None):
        if isinstance(X, list) or isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
            del X
        else:
            X_df = X.copy()
        # 填充缺失值(method='pad / ffill'表示用前一个值来填充缺失值;method='backfill / bfill'表示用后一个值来填充缺失值)
        # X_df.fillna(method='ffill', axis=0, inplace=True)
        X_df.fillna(0, axis=0, inplace=True)
        # 删除重复列(如果有)
        # X_df = utils_data_cleaning.drop_duplicate_columns(X_df)     # 一个python数据框中应该不存在两个相同的列名
        # 独热编码 XXX: 编码后线性坑
        # X_df = utils_data_cleaning.one_hot_encode(X_df)
        # print('编码后的特征大小', X_df.shape)

        # 如果y不为空，说明为训练集，进行特征选择，并保留选择的特征(预测时需要)
        clean_information_dict = {"scaler_fit": None, "feature_selection": None, "pca": None}
        if y is not None:
            if y.dtype == 'object':  # 如果不是数值型，就先进行label_encode
                y = utils_data_cleaning.label_encode(list(y))
            if self.scaler:
                mms_fit = MinMaxScaler().fit(X_df)
                X_df = mms_fit.transform(X_df)
                clean_information_dict["scaler_fit"] = mms_fit
            if self.feature_selection:
                feature_selsction = utils_feature_selection.\
                    FeatureSelectionTransformer(self.type_of_estimator, feature_selection_model='SelectFromModel')
                feature_selsction.fit(X_df, y)
                X_df, select_index = feature_selsction.transform(np.array(X_df))
                clean_information_dict["feature_selection"] = select_index
                # np.save('select_index.npy', select_index)
                print('特征选择后的大小：', X_df.shape)

            if self.pca:
                pca_fit = PCA(n_components=0.99).fit(X_df)
                # joblib.dump(pca_fit, "pca_fit.pkl")
                X_df = pca_fit.transform(X_df)
                clean_information_dict["pca"] = pca_fit
                print("提取{}个主成分".format(X_df.shape[1]))
            joblib.dump(clean_information_dict, "clean_information_dict.pkl")   # 2018-8-21将多个数据处理信息保存在一个字典中
            return X_df, y
        else:  # 否则y为空，说明为预测数据，选择特征选择后的数据
            X_df = np.array(X_df)
            clean_information_dict = joblib.load("clean_information_dict.pkl")
            if self.scaler:
                mms_fit = clean_information_dict['scaler_fit']
                if mms_fit is not None:
                    X_df = mms_fit.transform(X_df)
                else:
                    raise ValueError(" Value of scaler is 'None',can not transform X")
            if self.feature_selection:
                select_index = clean_information_dict['feature_selection']
                if select_index is not None:
                    X_df = X_df[:, select_index]
                else:
                    raise ValueError(" Value of select_index is 'None',can not transform X")
            if self.pca:
                pca_fit = clean_information_dict['pca']
                if pca_fit is not None:
                    X_df = pca_fit.transform(X_df)
                else:
                    raise ValueError(" Value of pca_fit is 'None',can not transform X")
            return X_df

    # 贝叶斯优化需要用到的评估函数
    def _evaluate(self, args):
        if self.type_of_estimator == 'classifier':
            scores = cross_validate(self.best_model(**args), self.X_df, self.y, scoring='accuracy',
                                    cv=self.cv_split, return_train_score=True, n_jobs=-1)
            test_accuracy = round(np.mean(scores['test_score']), 4)

            # return {'loss': -test_accuracy, 'status': STATUS_OK}
            return -test_accuracy

        elif self.type_of_estimator == 'regressor':
            scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
            scores = cross_validate(self.best_model(**args), self.X_df, self.y, scoring=scoring,
                                    cv=self.cv_split, return_train_score=True, n_jobs=-1)
            test_mae = round(np.mean(scores['test_neg_mean_absolute_error']), 4)

            return -test_mae
            # return {'loss': -test_r2, 'status': STATUS_OK}

    # 贝叶斯优化超参数
    def _optimise(self, max_evals=200):
        """
        :param max_evals: int, default = 200.
            Number of iterations.
            For an accurate optimal hyper-parameter, max_evals = 200.
        :return:dict.
            The optimal hyper-parameter dictionary.
        """

        hyper_space = {}
        for p in self.best_model_space.keys():
            if "space" not in self.best_model_space[p]:
                raise ValueError("You must give a space list ie values"
                                 " for hyper parameter " + p + ".")
            else:
                if "search" in self.best_model_space[p]:
                    if self.best_model_space[p]["search"] == "uniform":
                        hyper_space[p] = hp.uniform(p, np.sort(self.best_model_space[p]["space"])[0],  # noqa
                                                    np.sort(self.best_model_space[p]["space"])[-1])  # noqa

                    elif self.best_model_space[p]["search"] == "choice":
                        hyper_space[p] = hp.choice(p, self.best_model_space[p]["space"])
                    else:
                        raise ValueError(
                            "Invalid search strategy for hyper parameter " + p + ". Please"
                            " choose between 'choice' and 'uniform'.")
                else:
                    hyper_space[p] = hp.choice(p, self.best_model_space[p]["space"])
        # hyperopt_objective = lambda params: -self.evaluate(params)             # 删除
        best_params = fmin(self._evaluate,
                           space=hyper_space,
                           algo=tpe.suggest,
                           max_evals=max_evals)
        # Displaying best_params
        for p, v in best_params.items():
            if "search" in self.best_model_space[p]:
                if self.best_model_space[p]["search"] == "choice":
                    best_params[p] = self.best_model_space[p]["space"][v]
                else:
                    pass
            else:
                best_params[p] = self.best_model_space[p]["space"][v]

        return best_params

    # 训练数据集
    def train(self, raw_training_data, cv_n=5, feature_selection=False, pca=False):
        # cv_split = ShuffleSplit(n_splits=cv_n, test_size=0.2, train_size=0.8, random_state=66)  # 随机交叉验证函数
        cv_split = KFold(n_splits=cv_n, shuffle=True, random_state=66)                       # K折交叉验证函数  8-15
        self.cv_split = cv_split
        X_df, y = self.get_X_and_y(raw_training_data)
        self.feature_selection, self.pca = feature_selection, pca
        if self.feature_selection:
            print('原数据的特征大小', X_df.shape)
        X_df, y = self._clean_data(X_df, y)
        self.X_df = X_df
        self.y = y
        if self.type_of_estimator == 'classifier':
            # best_model, self.best_model_space = utils_models.run_classifiers(X_df, y, cv_split)
            # self.best_model = best_model
            # # 利用贝叶斯优化超参数，返回最好的参数
            # best_params = self._optimise()
            # best_scores = self._evaluate(args=best_params)   # 传入交叉验证参数 8-15
            # print("此分类模型测试集准确率:", -best_scores['loss'])
            # # 用得到的最优参数来更新best_model
            # final_best_model = best_model(**best_params)
            # print("最优参数为", final_best_model.get_params())
            # final_best_model.fit(X_df, y)
            # joblib.dump(final_best_model, "trained_best_model.pkl")

            # 2019-8-20修改为对前三个模型进行自动调优
            candidate_model_list, candidate_search_space_list = utils_models.run_classifiers(X_df, y, cv_split)
            print('分类模型的候选模型为：', candidate_model_list)
            all_params = []
            all_scores = []
            for i in range(len(candidate_model_list)):
                #
                self.best_model_space = candidate_search_space_list[i]
                self.best_model = candidate_model_list[i]
                print("{}模型调参中。。。。".format(self.best_model))
                # 利用贝叶斯优化超参数，返回最好的参数
                all_params.append(self._optimise())
                all_scores.append(self._evaluate(args=all_params[i]))

            best_model_index = np.argmin(all_scores)
            best_model = candidate_model_list[best_model_index]

            # 用得到的最优参数来更新best_model
            print("最优分类模型为:", best_model)
            print("此模型的准确率为:", -all_scores[best_model_index])
            final_best_model = best_model(**all_params[best_model_index])
            print("此模型最优参数为:", final_best_model.get_params())
            final_best_model.fit(X_df, y)
            # print(final_best_model.feature_importances_)
            joblib.dump(final_best_model, "trained_best_model.pkl")
            print("model saving completed ! !")

        elif self.type_of_estimator == 'regressor':
            # best_model, self.best_model_space = utils_models.run_regressions(X_df, y, cv_split)
            # self.best_model = best_model
            # print("最好的回归模型为", best_model)
            # # 利用贝叶斯优化超参数，返回最好的参数
            # best_params = self._optimise()
            # best_scores = self._evaluate(best_params)
            # print("此回归模型的MSE:", -best_scores['loss'])
            # # 用得到的最优参数来更新best_model
            # final_best_model = best_model(**best_params)
            # # print("最优参数为", final_best_model.get_params())
            # print("最优参数为", best_params)
            # final_best_model.fit(X_df, y)
            # joblib.dump(final_best_model, "trained_best_model.pkl")
            # 2019-8-20修改为对前三个模型进行自动调优
            candidate_model_list, candidate_search_space_list = utils_models.run_regressions(X_df, y, cv_split)
            print('回归模型的候选模型为：', candidate_model_list)
            all_params = []
            all_scores = []
            for i in range(len(candidate_model_list)):

                self.best_model_space = candidate_search_space_list[i]
                self.best_model = candidate_model_list[i]
                print("{}模型调参中。。。。".format(self.best_model))
                # 利用贝叶斯优化超参数，返回最好的参数
                all_params.append(self._optimise())
                all_scores.append(self._evaluate(args=all_params[i]))

            best_model_index = np.argmin(all_scores)
            best_model = candidate_model_list[best_model_index]

            # 用得到的最优参数来更新best_model
            print("最优回归模型为:", best_model)
            print("此模型的平均绝对误差为:", all_scores[best_model_index])
            final_best_model = best_model(**all_params[best_model_index])
            print("此模型最优参数为:", final_best_model.get_params())
            final_best_model.fit(X_df, y)
            # print(final_best_model.feature_importances_)
            joblib.dump(final_best_model, "trained_best_model.pkl")
            print("model saving completed ! !")

    # 预测新的数据
    def predict(self, predict_data):
        # 载入保存的模型
        model = joblib.load("trained_best_model.pkl")
        # 清洗数据并预测

        cleaned_predict_data = self._clean_data(predict_data)
        predict_y = model.predict(cleaned_predict_data)

        # 预测结果保存
        pd.Series(predict_y).to_csv("predict_out.csv", encoding='utf-8', index=False, mode='w')
        # if os.path.exists('predict_out.txt'):
        #     os.remove('predict_out.txt')
        # doc = open('predict_out.txt', 'a+')
        # for i in predict_y:
        #     doc.writelines(str(i))
        #     doc.write("\n")
        # doc.close()
        return predict_y
