import torch.nn as nn
import torch.nn.functional as F

class FaceMaskNet(nn.Module):
  """
  A model to detect facemasks
  References:
  (CNN) https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
  (Relu) https://www.superdatascience.com/blogs/convolutional-neural-networks-cnn-step-1b-relu-layer/
  (Calculating output channels) https://deeplizard.com/learn/video/cin4YcGBh3Q
  """

  def __init__(self):
    """
    Initialize a CNN model to detect facemask
    """
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2)
    self.fc1 = nn.Linear(in_features=4*4*64, out_features=64)
    self.softmax_layer = nn.Linear(in_features=64, out_features=2)

  def forward(self, X):
    """
    Forward pass of input through model 
    
    Params
    _________
    X: an input tensor to the CNN model
    """
    X = self.pool(F.relu(self.conv1(X)))
    X = self.pool(F.relu(self.conv2(X)))
    X = X.view(-1, 4*4*64)
    X = F.relu(self.fc1(X))
    return F.softmax(self.softmax_layer(X), dim=1)
