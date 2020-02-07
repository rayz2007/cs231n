{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "from cs231n.layers import *\n",
    "from cs231n.fast_layers import *\n",
    "from cs231n.layer_utils import *\n",
    "\n",
    "\n",
    "class ThreeLayerConvNet(object):\n",
    "  \"\"\"\n",
    "  A three-layer convolutional network with the following architecture:\n",
    "  \n",
    "  conv - relu - 2x2 max pool - affine - relu - affine - softmax\n",
    "  \n",
    "  The network operates on minibatches of data that have shape (N, C, H, W)\n",
    "  consisting of N images, each with height H and width W and with C input\n",
    "  channels.\n",
    "  \"\"\"\n",
    "  \n",
    "  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,\n",
    "               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,\n",
    "               dtype=np.float32):\n",
    "    \"\"\"\n",
    "    Initialize a new network.\n",
    "    \n",
    "    Inputs:\n",
    "    - input_dim: Tuple (C, H, W) giving size of input data\n",
    "    - num_filters: Number of filters to use in the convolutional layer\n",
    "    - filter_size: Size of filters to use in the convolutional layer\n",
    "    - hidden_dim: Number of units to use in the fully-connected hidden layer\n",
    "    - num_classes: Number of scores to produce from the final affine layer.\n",
    "    - weight_scale: Scalar giving standard deviation for random initialization\n",
    "      of weights.\n",
    "    - reg: Scalar giving L2 regularization strength\n",
    "    - dtype: numpy datatype to use for computation.\n",
    "    \"\"\"\n",
    "    self.params = {}\n",
    "    self.reg = reg\n",
    "    self.dtype = dtype\n",
    "    \n",
    "    ############################################################################\n",
    "    # TODO: Initialize weights and biases for the three-layer convolutional    #\n",
    "    # network. Weights should be initialized from a Gaussian with standard     #\n",
    "    # deviation equal to weight_scale; biases should be initialized to zero.   #\n",
    "    # All weights and biases should be stored in the dictionary self.params.   #\n",
    "    # Store weights and biases for the convolutional layer using the keys 'W1' #\n",
    "    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #\n",
    "    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #\n",
    "    # of the output affine layer.                                              #\n",
    "    ############################################################################\n",
    "    C, H, W = input_dim\n",
    "    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))\n",
    "    self.params['b1'] = np.zeros(num_filters)\n",
    "    self.params['W2'] = np.random.normal(0, weight_scale, (num_filters * H/2*W/2, hidden_dim))\n",
    "    self.params['b2'] = np.zeros(hidden_dim)\n",
    "    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))\n",
    "    self.params['b3'] = np.zeros(num_classes)\n",
    "    ############################################################################\n",
    "    #                             END OF YOUR CODE                             #\n",
    "    ############################################################################\n",
    "\n",
    "    for k, v in self.params.iteritems():\n",
    "      self.params[k] = v.astype(dtype)\n",
    "     \n",
    " \n",
    "  def loss(self, X, y=None):\n",
    "    \"\"\"\n",
    "    Evaluate loss and gradient for the three-layer convolutional network.\n",
    "    \n",
    "    Input / output: Same API as TwoLayerNet in fc_net.py.\n",
    "    \"\"\"\n",
    "    W1, b1 = self.params['W1'], self.params['b1']\n",
    "    W2, b2 = self.params['W2'], self.params['b2']\n",
    "    W3, b3 = self.params['W3'], self.params['b3']\n",
    "    \n",
    "    # pass conv_param to the forward pass for the convolutional layer\n",
    "    filter_size = W1.shape[2]\n",
    "    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}\n",
    "\n",
    "    # pass pool_param to the forward pass for the max-pooling layer\n",
    "    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}\n",
    "\n",
    "    scores = None\n",
    "    ############################################################################\n",
    "    # TODO: Implement the forward pass for the three-layer convolutional net,  #\n",
    "    # computing the class scores for X and storing them in the scores          #\n",
    "    # variable.                                                                #\n",
    "    ############################################################################\n",
    "    conv_out, conv_cache = conv_forward_im2col(X, W1, b1, conv_param)\n",
    "    conv_relu_out, conv_relu_cache = relu_forward(conv_out)\n",
    "    pool_out, pool_cache = max_pool_forward_fast(conv_relu_out, pool_param)\n",
    "    affine_relu_out, affine_relu_cache = affine_relu_forward(pool_out, W2, b2)\n",
    "    affine_out, affine_cache = affine_forward(affine_relu_out, W3, b3)\n",
    "    scores = affine_out\n",
    "    ############################################################################\n",
    "    #                             END OF YOUR CODE                             #\n",
    "    ############################################################################\n",
    "    \n",
    "    if y is None:\n",
    "      return scores\n",
    "    \n",
    "    loss, grads = 0, {}\n",
    "    ############################################################################\n",
    "    # TODO: Implement the backward pass for the three-layer convolutional net, #\n",
    "    # storing the loss and gradients in the loss and grads variables. Compute  #\n",
    "    # data loss using softmax, and make sure that grads[k] holds the gradients #\n",
    "    # for self.params[k]. Don't forget to add L2 regularization!               #\n",
    "    ############################################################################\n",
    "    loss, dscores = softmax_loss(scores, y)\n",
    "    loss += 0.5 * self.reg*(np.sum(self.params['W1']* self.params['W1']) + np.sum(self.params['W2']* self.params['W2'])+np.sum(self.params['W3']* self.params['W3']))\n",
    "    affine_dx3, affine_dw3, affine_db3 = affine_backward(dscores, affine_cache)\n",
    "    grads['W3'] = affine_dw3 + self.reg * self.params['W3']\n",
    "    grads['b3'] = affine_db3\n",
    "    affine_dx2, affine_dw2, affine_db2 = affine_relu_backward(affine_dx3, affine_relu_cache)\n",
    "    grads['W2'] = affine_dw2 + self.reg * self.params['W2']\n",
    "    grads['b2'] = affine_db2\n",
    "    pool_dx = max_pool_backward_fast(affine_dx2, pool_cache)\n",
    "    relu_dx = relu_backward(pool_dx, conv_relu_cache)\n",
    "    dx, dw, db = conv_backward_im2col(relu_dx, conv_cache)\n",
    "    grads['W1'] = dw + self.reg * self.params['W1']\n",
    "    grads['b1'] = db\n",
    "    ############################################################################\n",
    "    #                             END OF YOUR CODE                             #\n",
    "    ############################################################################\n",
    "    \n",
    "    return loss, grads\n",
    "  \n",
    "  \n",
    "pass\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Environment (conda_python2)",
   "language": "python",
   "name": "conda_python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
