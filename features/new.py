
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import stats

import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.constants import adjust_engagement,DATA_THESHOLD,read_data,DATA_FOLDER,IMPORTANT_COLUMNS,spearman_correlation,pearson_correlation,column_categories,plot_bar,plot_line


data_name='sales_data_sample.csv'

dataframe=read_data(data_name)
#print(dataframe.describe())

#column_names=dataframe.columns


objects_columns,numerics_columns=column_categories(dataframe)

print(objects_columns)
print(numerics_columns)

numeric_df=dataframe.select_dtypes(include=['int64','float64'])

correlation_matrix=numeric_df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


#plot_line(dataframe, 'QUANTITYORDERED', 'SALES')
#plot_line(dataframe, 'Discount', 'Profit')
plot_bar(dataframe,'PRODUCTCODE','SALES')
