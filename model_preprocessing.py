"""This module contains basic preprocessing before modelling.

Returns:
    None: Functions for Model Preprocessing.
"""
import numpy as np
import pandas as pd
import torch


def converts_to_category(data_frame: pd.DataFrame, features: list) -> None:
    """This function is for converting string categoriacal features to category datatypes.

    Args:
        data_frame (pd.DataFrame): The input Data Frame.
        features (list): The categorical features list.
    """   
    for col in features:
        data_frame[col] = data_frame[col].astype('category')     

def return_cats_tensor(data_frame: pd.DataFrame, cat_cols: list) -> torch.tensor:
    """This function returns categorical features as tensor. 

    Args:
        data_frame (pd.DataFrame): The input Data Frame.
        cat_cols (list): The categorical features list.

    Returns:
        torch.tensor: The 2 D categorical tensor.
    """    
    cats = np.stack([data_frame[col].cat.codes.values for col in cat_cols], axis=1)
    cats = torch.tensor(cats, dtype=torch.int64)
    return cats

def return_conts_tensor(data_frame: pd.DataFrame, cont_cols: list) -> torch.tensor:
    """This function returns continuous features as tensor.

    Args:
        data_frame (pd.DataFrame): The input Data Frame.
        cont_cols (list): The continuous features list.

    Returns:
        torch.tensor: The 2 D continuous tensor.
    """    
    conts = np.stack([data_frame[col].values for col in cont_cols], axis=1)
    conts = torch.tensor(conts, dtype=torch.float)
    return conts

def return_out_label(data_frame: pd.DataFrame, feature: str) -> torch.tensor:
    """This function returns the output tensor for classification.

    Args:
        data_frame (pd.DataFrame): The input Data Frame.
        feature (str): The name of the output feature.

    Returns:
        torch.tensor: _description_
    """    
    y_label = torch.tensor(data_frame[feature].values).flatten()
    return y_label
