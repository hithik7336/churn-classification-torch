"""This module contains basic metrics for classification.

Returns:
    None: Functions for Accuracy, Precision, Recall and Classification Report and Confusion Matrix.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score, recall_score)


def get_accuracy(predicted: np.array, original: np.array) -> float:
    """_summary_

    Args:
        predicted (np.array): _description_
        original (np.array): _description_

    Returns:
        float: _description_
    """    
    return accuracy_score()
