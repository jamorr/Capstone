import pandas as pd
import numpy as np

def rmse(df:pd.DataFrame, gt_lbl:str, pre_lbl:str):
    #TODO: seek DoF for better calc
    diff = (df[gt_lbl] - df[pre_lbl])
    return np.sqrt(np.dot(diff.values, diff.values)/ diff.shape[0])

