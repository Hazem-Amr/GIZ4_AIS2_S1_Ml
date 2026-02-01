import pandas as pd

def chk(df):
    dftypes = df.dtypes
    n_unique = df.nunique()
    return pd.DataFrame({"Dtypes" : dftypes,"Num_Unique" : n_unique}).T


#--------------------------------

