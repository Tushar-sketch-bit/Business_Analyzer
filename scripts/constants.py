import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pdfplumber
from scipy.stats import trim_mean
import wquantiles

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models')
GRAPHS_SAVED_PATH = os.path.join(BASE_DIR, 'outputs')

class FileHandler:
    """<h3>Handles file I/O operations like reading/writing files, handling errors, etc.</h3>"""
    def __init__(self, data_folder=DATA_FOLDER, graphs_path=GRAPHS_SAVED_PATH, model_path=MODEL_PATH):
        self.data_folder = data_folder
        self.graphs_path = graphs_path
        self.model_path = model_path
        self.last_loaded_df = None
        self.last_error = None

    def read_data(self, filename:str) -> pd.DataFrame:
        try:
            df = pd.read_csv(os.path.join(self.data_folder, filename), encoding='latin-1')
            self.last_loaded_df = df
            self.last_error = None
            return df
        except Exception as e:
            self.last_error = str(e)
            print(f"Error reading CSV: {e}")
            return None

    def write_data(self, df:pd.DataFrame, filename:str):
        try:
            df.to_csv(os.path.join(self.data_folder, filename), index=False, sep='\t')
            self.last_error = None
        except Exception as e:
            self.last_error = str(e)
            print(f"Error writing CSV: {e}")

    def read_pdf_table(self, file_path:str) -> pd.DataFrame:
        print(f"Reading PDF: {file_path}")
        try:
            with pdfplumber.open(file_path) as pdf:
                all_tables = []
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        all_tables.append(df)
                if all_tables:
                    result = pd.concat(all_tables, ignore_index=True)
                    self.last_loaded_df = result
                    self.last_error = None
                    return result
                else:
                    raise ValueError("No tables found in PDF.")
        except Exception as e:
            self.last_error = str(e)
            print(f"Error reading PDF: {e}")
            return None

    def agent_step1_load_data(self, file_path):
        try:
            if file_path.lower().endswith('.csv'):
                df = self.read_data(os.path.basename(file_path))
                if df is not None:
                    print("CSV loaded successfully.")
                return df
            elif file_path.lower().endswith('.pdf'):
                df = self.read_pdf_table(file_path)
                if df is not None:
                    print("PDF loaded successfully.")
                    return df
                else:
                    print("PDF could not be loaded.")
                    return None
            else:
                print("Unsupported file type.")
                self.last_error = "Unsupported file type."
                return None
        except Exception as e:
            self.last_error = str(e)
            print(f"Error loading file: {e}")
            return None

    def save_graph(self, graph_name):
        graph_path = os.path.join(self.graphs_path, graph_name)
        plt.savefig(graph_path)

    def show_saved_graphs(self, output_folder=None):
        folder = output_folder if output_folder else self.graphs_path
        for file in os.listdir(folder):
            if file.endswith('.png'):
                print(file)
                plt.figure()
                img = plt.imread(os.path.join(folder, file))
                plt.imshow(img)
                plt.axis('off')
                plt.show()

    def use_model(self, modelname:str):
        try:
            model = joblib.load(os.path.join(self.model_path, modelname))
            self.last_error = None
            return model
        except Exception as e:
            self.last_error = str(e)
            print(f"Error loading model: {e}")
            return None

    def adjust_engagement(self, val, DATA_THRESHOLD:int):
        return val * 0.7 if val > DATA_THRESHOLD else val

    def add_important_columns(self, name:str, important_columns:list=None) -> list:
        if important_columns is None:
            important_columns = []
        name = name.upper()
        important_columns.append(name)
        return important_columns

class CorrelationFeatures:
    def __init__(self, dataframe:pd.DataFrame=None):
        self.dataframe = dataframe
        self.object_columns = []
        self.numeric_columns = []
        if dataframe is not None:
            self.object_columns, self.numeric_columns = self.column_categories(dataframe)

    def pearson_correlation(self):
        if self.dataframe is not None:
            return self.dataframe.corr('pearson')
        return None

    def spearman_correlation(self):
        if self.dataframe is not None:
            return self.dataframe.corr('spearman')
        return None

    def column_categories(self, dataframe:pd.DataFrame=None) -> tuple:
        df = dataframe if dataframe is not None else self.dataframe
        if df is None:
            return [], []
        column_names = df.columns
        object_columns = [col for col in column_names if df[col].dtype == 'object']
        numeric_columns = [col for col in column_names if df[col].dtype != 'object']
        return object_columns, numeric_columns

    def get_important_correlations(self, threshold:float=0.6):
        if self.dataframe is None:
            return []
        corr_matrix = self.dataframe.corr(numeric_only=True)
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

    def pivot_products(self, index, column, value):
        if self.dataframe is not None:
            product_matrix = self.dataframe.pivot_table(index=index, columns=column, values=value).fillna(0)
            return product_matrix
        return None

class Eda:
    """Exploratory data analysis"""
    def __init__(self, dataframe:pd.DataFrame=None):
        self.dataframe = dataframe
        self.feature_columns = []

    def combine_two_features_multiply(self, feature1:str, feature2:str, new_feature:str) -> pd.DataFrame:
        """multiply Two features to create new feature \n
        Args: (dataframe,existing_feature1,existing_feature2,new_feature)"""
        if self.dataframe is not None:
            self.dataframe[new_feature] = self.dataframe[feature1] * self.dataframe[feature2]
            self.feature_columns.append(new_feature)
        return self.dataframe
    def combine_two_Features_add(dataframe:pd.DataFrame,feature1:str,feature2:str,new_feature:str)-> pd.DataFrame:
        """add Two features together to create new feature \n
           Args: (dataframe,existing_feature1,existing_feature2,new_feature)"""
        feature1=feature1.upper()
        feature2=feature2.upper()
        new_feature=new_feature.upper()
        dataframe[new_feature] = dataframe[feature1] + dataframe[feature2]
        return dataframe
    @staticmethod
    def combine_two_Features_minus(dataframe:pd.DataFrame,feature1:str,feature2:str,new_feature:str) -> pd.DataFrame:
        """Subtract to create new feature from existing Two features \n
             args: (dataframe,existing_feature1,existing_feature2,new_feature)"""
        feature1=feature1.upper()
        feature2=feature2.upper()
        new_feature=new_feature.upper()
        dataframe[new_feature] = dataframe[feature1] - dataframe[feature2]
        return dataframe
    
    @staticmethod
    def combine_two_Features_div(dataframe:pd.DataFrame,feature1:str,feature2:str,new_feature:str)-> pd.DataFrame:
        """Divide and create new feature from Two existing Features \n
         _ARGS_: (d)ataframe,existing_feature1,existing_feature2,new_feature)\n
         _type_: newfeature= feature1/feature2 \n
        Note: If there's division by zero, it will result in NaN values."""
        feature1=feature1.upper()
        feature2=feature2.upper()
        new_feature=new_feature.upper()
        dataframe[new_feature] = dataframe[feature1] / dataframe[feature2]
        return dataframe
    
    def clean_data_mod(self,column_name:str)-> pd.DataFrame:
        """Clean data by filling missing values with the mode of the column.\n
        Args: (dataframe,column_name)"""
        dataframe=self.dataframe
        dataframe[column_name]=dataframe[column_name].fillna(dataframe[column_name].mode()[0])
        return dataframe
    
    def clean_data_trimmean(self,column_name:str)-> pd.DataFrame:
        """Clean data by filling missing values with the mean of the column.\n
        Args: (dataframe,column_name)"""
        dataframe=self.dataframe
        trimmed_mean = trim_mean(dataframe[column_name], proportiontocut=0.1, axis=0)
        dataframe[column_name]=dataframe[column_name].fillna(trimmed_mean)
        return dataframe
    
    def middle_point_of_weighted_data(self,column_name:str,weight_column:str)-> pd.DataFrame:
        """calculate the weighted median of a column based on another column.\n
        Args: (dataframe,column_name,weight_column)"""
        dataframe=self.dataframe
        return wquantiles.median(dataframe[column_name], weights=dataframe[weight_column])
    
    def mode_per_product(self,subject:str,on_thebasis_of:str)-> pd.DataFrame:
        """calculate the mode of a column for each unique value in another column.\n
        Args: (dataframe,column_name)"""
        dataframe=self.dataframe
        return dataframe.groupby(subject)[on_thebasis_of].agg(lambda x: x.mode()[0])
    @staticmethod
    def get_mad(commodity:pd.DataFrame )-> float:
        """Get the mean abs deviation of a column or a commodity.\n
        Args:(commodity): dataframe or series \n
        Returns: Float value of spreadness of data from its mean value"""
        mean=commodity.mean()
        mad=np.sum(np.abs(commodity - mean))/len(commodity)
        return mad
        
    
class Visualization:
    """<H2>Plot line graphs, bar plots, and heatmaps.</H2>
    """
    @staticmethod
    def plot_line(dataframe, x_col, y_col):
        plt.figure(figsize=(10, 5))
        plt.plot(dataframe[x_col], dataframe[y_col], marker='o', linestyle='-')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{y_col} vs {x_col}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_bar(dataframe, category_col, numeric_col):
        plt.figure(figsize=(15, 6))
        sns.barplot(x=category_col, y=numeric_col, data=dataframe)
        plt.xlabel(category_col)
        plt.ylabel(numeric_col)
        plt.title(f"Bar Plot: {numeric_col} by {category_col}")
        plt.xticks(rotation=90)
        plt.show()

    @staticmethod
    def correlation_graph_plot(numeric_df:pd.DataFrame):
        correlation_matrix = numeric_df.corr()
        print(correlation_matrix)
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_pivot_products(products_df:pd.DataFrame, title):
        product_corr = products_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(product_corr, cmap='coolwarm', annot=True, linewidths=0.5)
        plt.title(f"{title}")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def product_product_corr_heatmap(df:pd.DataFrame):
        matrix = df.pivot_table(index='ORDERNUMBER', columns='PRODUCTCODE', values='SALES').fillna(0)
        corr = matrix.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap='coolwarm', annot=True, linewidths=0.5, center=0)
        plt.title('Product-Product Correlation Heatmap (sales)')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def numeric_target_corr_bar(df, target='SALES'):
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        corr = numeric_df.corr()[target].sort_values(ascending=False)
        corr.drop(target, inplace=True)
        corr.plot(kind='barh', figsize=(10, 6), color='green')
        plt.title(f'Correlation with {target}')
        plt.xlabel('Correlation Coefficients')
        plt.ylabel('Features')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
class main:
 def __init__(self,filename:str ,threshold: float,) -> None :
     """initialize the class instance with filename and threshold""" 
     self.filename = filename
     self.df = self.agent_step1_load_data(self.filename)
     self.DATA_THRESHOLD = threshold
     
     
     if self.df is not None:
         self.valid=True
         self.object_cols, self.num_cols = CorrelationFeatures.column_categories(self.df)
         print(f"[INIT] data loaded. {len(self.df)} rows. Numeric: {len(self.num_cols)}, categorical cols: {len(self.object_cols)}")
     else:
         self.valid=False
         self.object_cols,self.num_cols=None,None
         print("[INIT] invalid file path provided. Please provide valid csv/pdf file name]")   
         

if __name__=='__main__':
    filename="Sample_Data.csv"
    threshold=0.3
    obj=main(filename,threshold)
    print(obj.valid)
