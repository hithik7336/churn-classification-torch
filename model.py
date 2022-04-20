"""This module contains TabularModel Class.

Returns:
    None: The Tabular Model Class.
"""

from matplotlib import markers
import torch
from torch import embedding
from torch import nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from data_viz import go, pd, px


class TabularModel(nn.Module):
    """This class is for Tabular Model.

    Args:
        nn (nn.Module): The base class from which TabularModel inherits.
    """    
    def __init__(self, emb_szs: nn.ModuleList, n_conts: int, out_sz: int, layers: list, p: float=0.5) -> None:
        """This is the __init__ method of the TabularModel Class.

        Args:
            emb_szs (nn.ModuleList): The Embeddings used in categorical features.
            n_conts (int): Number of Continuous features, 
            out_sz (_type_): Number of output features.
            layers (_type_): Number of Neurons in each layer.
            p (float, optional): The % of neurons to dropped. Defaults to 0.5.
        """        
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.bn_cont = nn.BatchNorm1d(n_conts)
        self.dropout = nn.Dropout(p)

        layer_list = []
        n_embs = sum([nf for ni, nf in emb_szs])
        n_in = n_embs + n_conts

        for i in layers:
            layer_list.append(nn.Linear(n_in, i))
            layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.BatchNorm1d(i))
            layer_list.append(nn.Dropout(p))
            n_in = i

        layer_list.append(nn.Linear(layers[-1], out_sz))
        self.layers = nn.Sequential(*layer_list)
    
    def forward(self, x_cats: torch.tensor, x_conts:torch.tensor) -> torch.tensor:
        """This method used for applying forward propogation.

        Args:
            x_cats (torch.tensor): The categorical training set.
            x_conts (torch.tensor): The continuous trainig set.

        Returns:
            torch.tensor: The value as a tensor after forward pass.
        """        
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cats[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.dropout(x)

        x_conts = self.bn_cont(x_conts)
        x = torch.cat([x, x_conts], 1)
        x = self.layers(x)
        return x
    
def visualize_loss_per_epoch(epochs: list, loss_per_epoch: list) -> go.Figure:
    """This function is for plotting the Loss per Epoch.

    Args:
        epochs (list): The epochs as a list.
        loss_per_epoch (list): The loss at each epoch as a list. 

    Returns:
        go.Figure: Lineplot of Epochs and Loss per epoch.
    """    
    trace = [go.Scatter(x=epochs, y=loss_per_epoch, mode='lines+markers',
                        marker=dict(color='Red'))]
    layout = go.Layout(xaxis=dict(title='Epochs'), yaxis=dict(title='Loss'),
                       title='Loss Per Epoch')
    fig = go.Figure(data=trace, layout=layout)
    return fig

def prediction_on_test_data(model: TabularModel, cat_test: torch.tensor, cont_test: torch.tensor) -> torch.tensor:
    """This function is for getting predictions on test categorical and continuous tensors. 

    Args:
        model (TabularModel): The TabularModel object.
        cat_test (torch.tensor): The categorical test tensor.
        cont_test (torch.tensor): The continuous test tensor.

    Returns:
        torch.tensor: Output predictions as a tensor.
    """    
    with torch.no_grad():
        y_eval = model.forward(cat_test, cont_test)
        
    return y_eval
    