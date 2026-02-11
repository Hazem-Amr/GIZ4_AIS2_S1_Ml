import pandas as pd

def Nulls_chking(df):
    null = df.isnull().sum()
    ratio = (null / df.shape[0])*100
    return pd.DataFrame({"Nulls" : null,"Ratio" : ratio}).T