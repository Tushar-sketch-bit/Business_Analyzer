
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import stats, trim_mean
import pandas as pd
import os
import sys
import wquantiles
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.constants import FileHandler, CorrelationFeatures, Visualization, Eda

data_name = 'sales_data_sample.csv'  # or provide a PDF file path for PDF loading

# Use the new class-based FileHandler
file_handler = FileHandler()
dataframe = file_handler.agent_step1_load_data(data_name)

if dataframe is not None:
    print("Data loaded successfully. Shape:", dataframe.shape)
else:
    print("Failed to load data.")

# Use CorrelationFeatures as a stateful object
corr_features = CorrelationFeatures(dataframe)
object_cols, numeric_cols = corr_features.column_categories()

numeric_df = dataframe.select_dtypes(include=['int64', 'float64'])
Visualization.correlation_graph_plot(numeric_df)

important_correlations = corr_features.get_important_correlations(threshold=0.6)
for col1, col2, corr in important_correlations:
    print(f'{col1} and {col2}: Correlation: {corr}')

product_sales = dataframe.groupby('PRODUCTCODE')['SALES'].sum().reset_index()
filtered_product_sales = product_sales[product_sales['SALES'] > 8000]
print(filtered_product_sales.head(5))
filtered_product_sales.plot(kind='bar', x='PRODUCTCODE', y='SALES')
plt.xlabel('Product Line')
plt.ylabel('Total Sales')
plt.title('Top 3 Product Lines by Total Sales')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Uncomment if you want to use the pivot and heatmap features
# product_matrix = corr_features.pivot_products(index='ORDERNUMBER', column='PRODUCTCODE', value='SALES')
# print(product_matrix)
# Visualization.product_product_corr_heatmap(product_matrix)

# Use Eda as a stateful object for feature engineering
eda = Eda(dataframe)
dataframe = eda.combine_two_Features_minus('msrp', 'priceeach', 'discount')
dataframe = eda.combine_two_Features_div('discount', 'msrp', 'discount_percentage')
dataframe = eda.combine_two_features_multiply('msrp', 'quantityordered', 'totalrevenue')
dataframe['DISCOUNT_PERCENTAGE'] = dataframe['DISCOUNT_PERCENTAGE'].clip(lower=0)
dataframe['DISCOUNT_PERCENTAGE'] = dataframe['DISCOUNT_PERCENTAGE'] * 100
print(dataframe[['DISCOUNT', 'DISCOUNT_PERCENTAGE', 'TOTALREVENUE']].head())

print(dataframe.isnull().sum())
dataframe['ADDRESSLINE2'] = dataframe['ADDRESSLINE2'].fillna("Default address")
dataframe['STATE'] = dataframe['STATE'].fillna(dataframe['CITY'])
dataframe['POSTALCODE'] = dataframe['POSTALCODE'].fillna(dataframe['POSTALCODE'].mode()[0])
dataframe["TERRITORY"] = dataframe["TERRITORY"].fillna(dataframe["CITY"])
print(dataframe.isnull().sum())

trimmed_mean = trim_mean(dataframe['DISCOUNT'], proportiontocut=0.1, axis=0)
mean = np.mean(dataframe['DISCOUNT'])
median = np.median(dataframe['DISCOUNT'])
weighted_median = wquantiles.median(dataframe['DISCOUNT'], weights=dataframe['TOTALREVENUE'])
print(f"normal mean of discount{mean} \n trimmed mean(0.1) on discount{trimmed_mean}) \n median({median}) \n weighted median ({weighted_median})")

# Mode quantity ordered per product
mode_quantity = dataframe.groupby('PRODUCTCODE')['QUANTITYORDERED'].agg(lambda x: x.mode()[0]).reset_index()
mode_quantity.columns = ['PRODUCTCODE', 'MODE_QUANTITY']

# Merge back
dataframe = dataframe.merge(mode_quantity, on='PRODUCTCODE', how='left')
