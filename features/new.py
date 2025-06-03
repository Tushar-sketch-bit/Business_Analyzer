
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import stats
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.constants import FileHandler, CorrelationFeatures, Visualization,Eda

data_name = 'sales_data_sample.csv'  # or provide a PDF file path for PDF loading

dataframe = FileHandler.agent_step1_load_data(data_name)

if dataframe is not None:
    print("Data loaded successfully. Shape:", dataframe.shape)
else:
    print("Failed to load data.")

object_cols,numeric_cols=CorrelationFeatures.column_categories(dataframe)



numeric_df=dataframe.select_dtypes(include=['int64','float64'])
Visualization.correlation_graph_plot(numeric_df)


important_correlations=CorrelationFeatures.get_important_correlations(dataframe,threshold=0.6)
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



#product_matrix=CorrelationFeatures.pivot_products(dataframe,index='ORDERNUMBER',column='PRODUCTCODE',value='SALES')
#print(product_matrix)
#Visualization.product_product_corr_heatmap(product_matrix)

#df=CorrelationFeatures.combine_two_features_multiply(dataframe,'QUANTITYORDERED','PRICEEACH')
#print(df['QUANTITYORDERED_PRICEEACH'].head(5))

print(dataframe.isnull().sum())
dataframe['ADDRESSLINE2']=dataframe['ADDRESSLINE2'].fillna("Default address")
dataframe['STATE']=dataframe['STATE'].fillna(dataframe['CITY'])
dataframe['POSTALCODE']=dataframe['POSTALCODE'].fillna(dataframe['POSTALCODE'].mode()[0])
dataframe["TERRITORY"]=dataframe["TERRITORY"].fillna(dataframe["CITY"])
print(dataframe.isnull().sum())

dataframe=Eda.combine_two_Features_minus(dataframe,'msrp','priceeach','discount')
dataframe=Eda.combine_two_Features_div(dataframe,'discount','msrp','discount_percentage')
dataframe=Eda.combine_two_features_multiply(dataframe,'msrp','quantityordered','totalrevenue')
dataframe['DISCOUNT_PERCENTAGE'] = dataframe['DISCOUNT_PERCENTAGE'].clip(lower=0)

print(dataframe[['DISCOUNT', 'DISCOUNT_PERCENTAGE', 'TOTALREVENUE']].head())
