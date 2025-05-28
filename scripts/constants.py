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
def use_model(modelname):
    return joblib.load(os.path.join(MODEL_PATH,modelname)) 


DATA_THESHOLD=500

def adjust_engagement(val):
    return val*0.7 if val>DATA_THESHOLD else val  
        
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