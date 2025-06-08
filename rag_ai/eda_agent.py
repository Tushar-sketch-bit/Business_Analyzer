from scripts.constants import Eda
from rag_ai.data_load_agent import DataLoadAgent

class EdaAgent:
    """Agent for performing exploratory data analysis on the loaded data."""
    def __init__(self, data_load_agent: DataLoadAgent):
        self.data_load_agent = data_load_agent
        self.eda = Eda(dataframe=data_load_agent.get_dataframe())

    def run(self):
        try:
            # Perform EDA using methods from constants.py
            self.eda.middle_point_of_weighted_data('DISCOUNT', 'TOTALREVENUE')
            self.eda.mode_per_product('PRODUCTCODE', 'ORDERQUANTITY')
            self.eda.clean_data_trimmean('DISCOUNT')
            self.eda.dataframe = Eda.combine_two_Features_minus(self.eda.dataframe, 'msrp', 'priceeach', 'discount')
            
            # These are static methods, so we need to pass the dataframe
            self.eda.dataframe = Eda.combine_two_Features_div(self.eda.dataframe, 'discount', 'msrp', 'discount_percentage')
            self.eda.combine_two_features_multiply('msrp', 'quantityordered', 'totalrevenue')

            # Get the updated dataframe after EDA
            self.dataframe = self.eda.dataframe

            # Print the results of EDA
            print("EDA Results:")
            print(self.dataframe.head())

        except Exception as e:
            print(f"Error in EdaAgent: {e}")