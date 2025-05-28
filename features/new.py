from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import math
import pandas as pd
import kagglehub 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.constants import adjust_engagement,DATA_THESHOLD,read_data,DATA_FOLDER,IMPORTANT_COLUMNS,spearman_correlation,pearson_correlation

data_name='sales_data_sample.csv'

dataframe=read_data(data_name)
#print(dataframe.describe())

#column_names=dataframe.columns

def column_categories(dataframe):
    column_names=dataframe.columns
    OBJECT_COLUMNS=[]
    NUMERIC_COLUMNS=[]
    for column in column_names:
        if dataframe[column].dtype=='object':
            OBJECT_COLUMNS.append(column)
        continue
    else:
        pass

#Numeric columns are those which have int or float datatype.

    for column in column_names:
      if dataframe[column].dtype!='object':
        NUMERIC_COLUMNS.append(column)
        continue
    else:
        pass
        return OBJECT_COLUMNS,NUMERIC_COLUMNS

objects_columns,numerics_columns=column_categories(dataframe)

#print(objects_columns)


numeric_df=dataframe.select_dtypes(include=['int64','float64'])

correlation_matrix=numeric_df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
