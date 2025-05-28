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
        
        
        
def pearson_correlation(filename):
     df=pd.read_csv(os.path.join(DATA_FOLDER,filename),encoding='latin-1')
     return df.corr('pearson')
 
def spearman_correlation(filename):
     df=pd.read_csv(os.path.join(DATA_FOLDER,filename),encoding='latin-1')
     return df.corr('spearman') 