import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

BASE_DIR=os.path.dirname(os.path.dirname(__file__))
DATA_FOLDER=os.path.join(BASE_DIR,'data')
MODEL_PATH=os.path.join(BASE_DIR,'models')
GRAPHS_SAVED_PATH=os.path.join(BASE_DIR,'outputs')


#Data utils
IMPORTANT_COLUMNS=[]
def read_data(filename):
    return pd.read_csv(os.path.join(DATA_FOLDER,filename),encoding='latin-1')

def write_data(df,filename):
    df.to_csv(os.path.join(DATA_FOLDER,filename),index=False,sep='\t')
    
def save_graph(graph_name):
    graph_path=os.path.join(GRAPHS_SAVED_PATH,graph_name)
    plt.savefig(graph_path)    
    
def show_saved_graphs(output_folder):
    for file in os.listdir(output_folder):
        if file.endswith('.png'):
            print(file)
            plt.figure()
            img=plt.imread(os.path.join(output_folder,file))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
  
#Model utils           
def use_model(modelname):
    return joblib.load(os.path.join(MODEL_PATH,modelname)) 

def adjust_engagement(val,DATA_THRESHOLD):
    return val*0.7 if val>DATA_THRESHOLD else val  


#Data Analysis utils
def add_important_columns(name):
    return IMPORTANT_COLUMNS.append(name)       
def pearson_correlation(filename):
     df=pd.read_csv(os.path.join(DATA_FOLDER,filename),encoding='latin-1')
     return df.corr('pearson')
 
def spearman_correlation(filename):
     df=pd.read_csv(os.path.join(DATA_FOLDER,filename),encoding='latin-1')
     return df.corr('spearman') 
 
 
def column_categories(dataframe):
    column_names=dataframe.columns
    OBJECT_COLUMNS=[]
    NUMERIC_COLUMNS=[]
    for column in column_names:
        if dataframe[column].dtype=='object':
            OBJECT_COLUMNS.append(column)
        continue
    else:
        #print(f"{column} is Numeric")
        pass

 #Numeric columns are those which have int or float datatype.

    for column in column_names:
      if dataframe[column].dtype!='object':
        NUMERIC_COLUMNS.append(column)
        continue
    else:
        #print(f"{column} is Categorical")
        pass
        return OBJECT_COLUMNS,NUMERIC_COLUMNS
    
def get_important_correlations(df, threshold):
    corr_matrix = df.corr(numeric_only=True)
    strong_corrs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                strong_corrs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_value
                ))

    strong_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
    return strong_corrs

def pivot_products(dataframe,index,column,value):
    product_matrix=dataframe.pivot_table(index=index,columns=column,values=value).fillna(0)
    return product_matrix 
        
#Graph utils
def plot_line(dataframe,x_col,y_col):
    plt.figure(figsize=(10,5))
    plt.plot(dataframe[x_col],dataframe[y_col],marker='o',linestyle='-')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_bar(dataframe,category_col,numeric_col):
    plt.figure(figsize=(15, 6))
    sns.barplot(x=category_col, y=numeric_col, data=dataframe)
    plt.xlabel(category_col)
    plt.ylabel(numeric_col)
    plt.title(f"Bar Plot: {numeric_col} by {category_col}")
    plt.xticks(rotation=90)
    plt.show()    
    
def correlation_graph_plot(numeric_df):
    correlation_matrix=numeric_df.corr()
    print(correlation_matrix)
    plt.figure(figsize=(10,8))
    sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_pivot_products(products_df,title):
    product_corr=products_df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(product_corr,cmap='coolwarm',annot=True,linewidths=0.5)
    plt.title(f"{title}")
    plt.tight_layout()
    plt.show()

def product_product_corr_heatmap(df):
    matrix=df.pivot_table(index='ORDERNUMBER',columns='PRODUCTCODE',values='SALES').fillna(0)
    corr=matrix.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr,cmap='coolwarm',annot=True,linewidths=0.5,center=0)
    plt.title('Product-Product Correlation Heatmap (sales)')
    plt.tight_layout()
    plt.show()

def numeric_target_corr_bar(df,target='SALES'):
    numeric_df=df.select_dtypes(include=['int64','float64'])
    corr=numeric_df.corr()[target].sort_values(ascending=False)
    corr.drop(target,inplace=True)
    corr.plot(kind='barh',figsize=(10,6),color='green')
    plt.title(f'Correlation with {target}')
    plt.xlabel('Correlation Coefficients')
    plt.ylabel('Features')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def numeric_target_corr_bar(df,target): 
    numeric_df=df.select_dtypes(include=['int64','float64'])
    corr=numeric_df.corr()[target].sort_values(ascending=False)
    corr.drop(target,inplace=True)
    corr.plot(kind='barh',figsize=(10,6),color='green')
    plt.title(f'Correlation with {target}')
    plt.xlabel('Correlation Coefficients')
    plt.ylabel('Features')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
