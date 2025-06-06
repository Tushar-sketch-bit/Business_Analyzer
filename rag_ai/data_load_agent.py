import os
from scripts.constants import agent_step1_load_data
import pandas as pd
import pdfplumber
class DataLoadAgent:
    """Agent for step 1: loading CSV or PDF files with error handling."""
    def __init__(self, file_path:pd.DataFrame| pdfplumber.pdf.PDF):
        self.file_path = file_path
        self.dataframe = None
        self.status = None
        self.error = None

    def run(self):
        try:
            self.dataframe = agent_step1_load_data(self.file_path)
            if self.dataframe is not None:
                self.status = "success"
                print(f"Data loaded successfully from {self.file_path}")
            else:
                self.status = "fail"
                print(f"Failed to load data from {self.file_path}")
        except Exception as e:
            self.status = "error"
            self.error = str(e)
            print(f"Error in DataLoadAgent: {e}")

    def get_dataframe(self):
        return self.dataframe

    def get_status(self):
        return self.status

    def get_error(self):
        return self.error