
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import stats
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.constants import FileHandler, CorrelationFeatures, Visualization

data_name = 'sales_data_sample.csv'  # or provide a PDF file path for PDF loading

dataframe = FileHandler.agent_step1_load_data(data_name)

if dataframe is not None:
    print("Data loaded successfully. Shape:", dataframe.shape)
else:
    print("Failed to load data.")

object_cols,numeric_cols=column_categories(dataframe)



numeric_df=dataframe.select_dtypes(include=['int64','float64'])
correlation_graph_plot(numeric_df)


important_correlations=get_important_correlations(dataframe,threshold=0.6)
for col1,col2, corr in important_correlations:
    print(f'{col1} and {col2}: Correlation: {corr}')

product_sales=dataframe.groupby('PRODUCTCODE')['SALES'].sum().reset_index()

filtered_product_sales=product_sales[product_sales['SALES']>8000]
print(filtered_product_sales.head(5))
filtered_product_sales.plot(kind='bar',x='PRODUCTCODE',y='SALES')
plt.xlabel('Product Line')
plt.ylabel('Total Sales')
plt.title('Top 3 Product Lines by Total Sales')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()



product_matrix=pivot_products(dataframe,index='ORDERNUMBER',column='PRODUCTCODE',value='SALES')
print(product_matrix)