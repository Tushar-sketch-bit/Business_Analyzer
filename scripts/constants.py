import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pdfplumber

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models')
GRAPHS_SAVED_PATH = os.path.join(BASE_DIR, 'outputs')

IMPORTANT_COLUMNS = []

class FileHandler:
    @staticmethod
    def read_data(filename):
        return pd.read_csv(os.path.join(DATA_FOLDER, filename), encoding='latin-1')

    @staticmethod
    def write_data(df, filename):
        df.to_csv(os.path.join(DATA_FOLDER, filename), index=False, sep='\t')

    @staticmethod
    def read_pdf_table(file_path):
        """Reads tables from a PDF file and returns a pandas DataFrame. Handles errors gracefully."""
        try:
            with pdfplumber.open(file_path) as pdf:
                all_tables = []
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        all_tables.append(df)
                if all_tables:
                    return pd.concat(all_tables, ignore_index=True)
                else:
                    raise ValueError("No tables found in PDF.")
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return None

    @staticmethod
    def agent_step1_load_data(file_path):
        """Agent step 1: Loads CSV or PDF as DataFrame with error handling."""
        try:
            if file_path.lower().endswith('.csv'):
                df = FileHandler.read_data(os.path.basename(file_path))
                print("CSV loaded successfully.")
                return df
            elif file_path.lower().endswith('.pdf'):
                df = FileHandler.read_pdf_table(file_path)
                if df is not None:
                    print("PDF loaded successfully.")
                    return df
                else:
                    print("PDF could not be loaded.")
                    return None
            else:
                print("Unsupported file type.")
                return None
        except Exception as e:
            print(f"Error loading file: {e}")
            return None

    @staticmethod
    def save_graph(graph_name):
        graph_path = os.path.join(GRAPHS_SAVED_PATH, graph_name)
        plt.savefig(graph_path)

    @staticmethod
    def show_saved_graphs(output_folder):
        for file in os.listdir(output_folder):
            if file.endswith('.png'):
                print(file)
                plt.figure()
                img = plt.imread(os.path.join(output_folder, file))
                plt.imshow(img)
                plt.axis('off')
                plt.show()

    @staticmethod
    def use_model(modelname):
        return joblib.load(os.path.join(MODEL_PATH, modelname))

    @staticmethod
    def adjust_engagement(val, DATA_THRESHOLD):
        return val * 0.7 if val > DATA_THRESHOLD else val

    @staticmethod
    def add_important_columns(name):
        return IMPORTANT_COLUMNS.append(name)

class CorrelationFeatures:
    @staticmethod
    def pearson_correlation(filename):
        df = FileHandler.read_data(filename)
        return df.corr('pearson')

    @staticmethod
    def spearman_correlation(filename):
        df = FileHandler.read_data(filename)
        return df.corr('spearman')

    @staticmethod
    def column_categories(dataframe):
        column_names = dataframe.columns
        OBJECT_COLUMNS = []
        NUMERIC_COLUMNS = []
        for column in column_names:
            if dataframe[column].dtype == 'object':
                OBJECT_COLUMNS.append(column)
        for column in column_names:
            if dataframe[column].dtype != 'object':
                NUMERIC_COLUMNS.append(column)
        return OBJECT_COLUMNS, NUMERIC_COLUMNS

    @staticmethod
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

    @staticmethod
    def pivot_products(dataframe, index, column, value):
        product_matrix = dataframe.pivot_table(index=index, columns=column, values=value).fillna(0)
        return product_matrix
    
    
class Eda:
    @staticmethod
    def combine_two_features_multiply(dataframe,feature1,feature2,new_feature):
        """Multiply two features together to create a new feature"""
        """args: (dataframe,existing_feature1,existing_feature2,new_feature)"""
        feature1=str(feature1).upper()
        feature2=str(feature2).upper()
        new_feature=str(new_feature).upper()
        dataframe[new_feature]=dataframe[feature1]*dataframe[feature2]
        return dataframe
    
    @staticmethod
    def combine_two_Features_add(dataframe,feature1,feature2,new_feature):
        """add Two features together to create new feature"""
        """args: (dataframe,existing_feature1,existing_feature2,new_feature)"""
        feature1=str(feature1).upper()
        feature2=str(feature2).upper()
        new_feature=str(new_feature).upper()
        dataframe[new_feature] = dataframe[feature1] + dataframe[feature2]
        return dataframe
    @staticmethod
    def combine_two_Features_minus(dataframe,feature1,feature2,new_feature):
        """Sbtract to create new feature from existing Two features"""
        """args: (dataframe,existing_feature1,existing_feature2,new_feature)"""
        feature1=str(feature1).upper()
        feature2=str(feature2).upper()
        new_feature=str(new_feature).upper()
        dataframe[new_feature] = dataframe[feature1] - dataframe[feature2]
        return dataframe
    
    @staticmethod
    def combine_two_Features_div(dataframe,feature1,feature2,new_feature):
        """Divide and create new feature from Two existing Features"""
        """args: (dataframe,existing_feature1,existing_feature2,new_feature)"""
        feature1=str(feature1).upper()
        feature2=str(feature2).upper()
        new_feature=str(new_feature).upper()
        dataframe[new_feature] = dataframe[feature1] / dataframe[feature2]
        return dataframe
    
    
class Visualization:
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
    def correlation_graph_plot(numeric_df):
        correlation_matrix = numeric_df.corr()
        print(correlation_matrix)
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_pivot_products(products_df, title):
        product_corr = products_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(product_corr, cmap='coolwarm', annot=True, linewidths=0.5)
        plt.title(f"{title}")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def product_product_corr_heatmap(df):
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
    
