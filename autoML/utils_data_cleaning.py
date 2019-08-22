import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
from pandas import get_dummies
import numpy as np



class CustomLabelEncoder:

    def __init__(self):
        self.label_map = {}

    def fit(self, list_of_labels):
        if not isinstance(list_of_labels, pd.Series):
            list_of_labels = pd.Series(list_of_labels)
        unique_labels = list_of_labels.unique()
        try:
            unique_labels = sorted(unique_labels)
        except TypeError:
            unique_labels = unique_labels

        for idx, val in enumerate(unique_labels):
            self.label_map[val] = idx
        return self

    def transform(self, in_values):
        return_values = []
        for val in in_values:
            if not isinstance(val, str):
                if isinstance(val, float) or isinstance(val, int) or val is None or isinstance(
                        val, np.generic):
                    val = str(val)
                else:
                    val = val.encode('utf-8').decode('utf-8')

            if val not in self.label_map:
                self.label_map[val] = len(self.label_map.keys())
            return_values.append(self.label_map[val])

        if len(in_values) == 1:
            return return_values[0]
        else:
            return return_values
    def fit_transform(self,y):
        self.fit(y)
        encoder_y = self.transform(y)
        return encoder_y


def one_hot_encode(data):
    """Perform a one-hot encoding and return as pandas data frame."""

    return get_dummies(data)


def label_encode(y):
    le = LabelEncoder()
    y_le = le.fit_transform(y)
    return y_le

def drop_duplicate_columns(df):
    count_cols_to_drop = 0
    cols = list(df.columns)
    for idx, item in enumerate(df.columns):
        if item in df.columns[:idx]:
            print('#' * 64)
            print('We found a duplicate column, and will be removing it')
            print('If you intended to send in two different pieces of information, please make '
                  'sure they have different column names.')
            print('Here is the duplicate column:')
            print(item)
            print('#' * 64)
            cols[idx] = 'DROPME'
            count_cols_to_drop += 1

    if count_cols_to_drop > 0:
        df.columns = cols

        df.drop('DROPME', axis=1, inplace=True)
    return df
if __name__ == '__main__':
    # path = 'C:/Users/Administrator/Desktop/DataSets/Multiple classification/ecoli.csv'
    path = 'train.csv'
    data = pd.read_csv(path, encoding="ANSI")
    data.dropna(axis=0, inplace=True)
    pp = CustomLabelEncoder()
    y = pp.fit_transform(data.iloc[:,-1])

    print(y)


