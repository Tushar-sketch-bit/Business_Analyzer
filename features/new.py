from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import math
import pandas as pd
from scripts.constants import adjust_engagement,DATA_THESHOLD,read_data
import kagglehub 

path=kagglehub.dataset_download("kyanyoga/sample-sales-data")
print(path)

df=read_data('sample-sales-data')
df.describe()






