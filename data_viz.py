"""This module contains basic functions for data visualization.

Returns:
    None: Functions for Data Visualizations.
"""
import numpy as np
import pandas as pd
from matplotlib.pyplot import title
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objs as go


def creates_distplot(data: pd.DataFrame, feature: str, color: str) -> go.Figure:
    """This function is for creating a distplot on a 1 D array numbers.

    Args:
        data (pd.DataFrame): The input Data Frame.
        feature (str): The Numerical feature whose plot is to be created.
        color (str): Color of the plot.

    Returns:
        go.Figure: The Distplot of provided continuous feature.
    """    
    fig = ff.create_distplot(hist_data=[data[feature].values], 
                             group_labels=[feature.capitalize()], 
                             colors=[color])
    return fig

def creates_countplot(data: pd.DataFrame, feature: str, color: str) -> go.Figure:
    """This function is for creating a countplot of a categorical feature.

    Args:
        data (pd.DataFrame): The input Data Frame
        feature (str): The categorical feature name.
        color (str): Color of the Bar Plot.

    Returns:
        go.Figure: The Countplot of provided categorical feature.
    """    
    count_data = data[feature].value_counts()
    trace = [go.Bar(x=count_data.index, y=count_data.values, marker=dict(color=color))]
    layout = go.Layout(xaxis=dict(title=feature), 
                       yaxis=dict(title=feature+" "+"count"),
                       title=f'Countplot of {feature}')
    fig = go.Figure(data=trace, layout=layout)
    
    return fig

def creates_pie_aggregate(data: pd.DataFrame, cat_feature: str, cont_feature: str, agg_method) -> go.Figure:
    """This function is for creating pie plot with a categorical and continuous feature.

    Args:
        data (pd.DataFrame): The input Data Frame.
        cat_feature (str): The categorical feature.
        cont_feature (str): The continuous feature.
        agg_method (_type_): The aggregate method of your choice.

    Returns:
        go.Figure: The Pie Plot figure. 
    """    
    by_feature = data.groupby(cat_feature)
    cats = data[cat_feature].unique()
    vals = [agg_method(by_feature.get_group(val)[cont_feature]) for val in data[cat_feature].unique()]
    
    fig = go.Figure(data=[go.Pie(labels=cats, values=vals, title='Aggregate Pie Chart')])
    return fig
        
def creates_bar_aggregate(data: pd.DataFrame, cat_feature: str, cont_feature: str, agg_method) -> go.Figure:
    """This function is for creating bar plot with a categorical and continuous feature.

    Args:
        data (pd.DataFrame): The input Data Frame.
        cat_feature (str): The categorical feature.
        cont_feature (str): The continuous feature.
        agg_method (_type_): The aggregate method of your choice.

    Returns:
        go.Figure: The Bar Plot figure. 
    """    
    by_feature = data.groupby(cat_feature)
    cats = data[cat_feature].unique()
    vals = [agg_method(by_feature.get_group(val)[cont_feature]) for val in data[cat_feature].unique()]
    trace = [go.Bar(x=cats, y=vals, marker=dict(color='rgb(52, 34, 67)'))]
    layout = go.Layout(xaxis=dict(title=cat_feature), yaxis=dict(title=cont_feature),
                        title='Aggregate Barplot')
    fig = go.Figure(data=trace, layout=layout)
    return fig
