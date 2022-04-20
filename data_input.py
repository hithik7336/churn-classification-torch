"""This module contains basic functions for data input.

Returns:
    None: Functions for Data Inputs.
"""
import pandas as pd

def read_csv(path: str) -> pd.DataFrame:
    """This function is for reading CSV file.

    Args:
        path (str): Location of the CSV file.

    Returns:
        pd.DataFrame: CSV File loaded as Data Frame.
    """  
    data = pd.read_csv(path)
    return data  

def return_shape(data_frame: pd.DataFrame) -> tuple:
    """This function is for returning shape of the data.

    Args:
        data_frame (pd.DataFrame): The input Data Frame.

    Returns:
        tuple: Shape in tuple as (rows, columns).
    """    
    shape = data_frame.shape
    return shape

def return_info(data_frame: pd.DataFrame) -> pd.Series:
    """This function is for returning info about dataset.

    Args:
        data_frame (pd.DataFrame): The input Data Frame.

    Returns:
        pd.Series: A series returning info of data.
    """    
    info = data_frame.info()
    return info

def return_null_values(data_frame: pd.DataFrame) -> pd.Series:
    """This function is for returning sum of NULL value in each feature.

    Args:
        data_frame (pd.DataFrame): The input Data Frame.

    Returns:
        pd.Series: NULL Values Data Frame.
    """    
    null_values = data_frame.isna().sum()
    return null_values

def removes_indices(data_frame: pd.DataFrame, indices: list, axis: bool) -> None:
    """This function is for removing rows or columns from Data Frame.

    Args:
        data_frame (pd.DataFrame): The input Data Frame.
        indices (list): Indices or Names of the rows or columns to be removed.
        axis (bool): 0 for rows and 1 for columns.
    """    
    data_frame.drop(indices, axis=axis, inplace=True)
