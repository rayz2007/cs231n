import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *



class FirstConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
  
    def __init__(self, input_dim=(3, 32, 32), num_filters=[16, 32], filter_size=7, hidden_dims=[100, 100], num_classes=10, weight_scale=1e-3, reg=0.0, use_batchnorm=False,dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.filter_size = filter_size
        self.use_batchnorm = use_batchnorm
        self.num_conv_layers = len(num_filters)
        self.num_affine_layers = 1 + len(hidden_dims)
        C, H, W = input_dim
        for i in range(self.num_conv_layers):
            layer_depth = C if i == 0 else num_filters[i - 1]
            self.params['W' + str(i)] = np.random.normal(0, weight_scale, (num_filters[i], layer_depth, filter_size, filter_size))
            self.params['b' + str(i)] = np.zeros(num_filters[i])
            if self.use_batchnorm:
                    self.params['gamma' + str(i)] = np.ones(num_filters[i])
                    self.params['beta' + str(i)] = np.zeros(num_filters[i])
            H /= 2
            W /= 2

        dim = 0
        for i in range(self.num_conv_layers, self.num_conv_layers + self.num_affine_layers):
            layer_input_dim = (H*W*(num_filters[len(num_filters)-1])) if i == self.num_conv_layers else hidden_dims[dim-1]
            layer_output_dim = num_classes if dim == len(hidden_dims) else hidden_dims[dim]
            self.params['W' + str(i)] = np.random.normal(0, weight_scale, (layer_input_dim, layer_output_dim))
            self.params['b' + str(i)] = np.zeros(layer_output_dim)

            if self.use_batchnorm and i != (self.num_conv_layers + self.num_affine_layers -1):
                self.params['gamma' + str(i)] = np.ones(layer_output_dim)
                self.params['beta' + str(i)] = np.zeros(layer_output_dim)
            dim += 1

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(self.num_conv_layers + self.num_affine_layers - 1)]

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = self.filter_size
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        conv_cache = {}
        affine_cache = {}
        next_input = X
        for i in range(self.num_conv_layers):
            next_input, conv_cache[i] = conv_bn_relu_pool_forward(next_input, self.params['W' + str(i)], self.params['b' + str(i)], conv_param, pool_param, self.params['gamma' + str(i)], self.params['beta' + str(i)], self.bn_params[i])

        for i in range(self.num_conv_layers, self.num_conv_layers + self.num_affine_layers-1):
            next_input, affine_cache[i] = affine_bn_relu_forward(next_input, self.params['W' + str(i)], self.params['b' + str(i)], self.params['gamma' + str(i)], self.params['beta' + str(i)], self.bn_params[i])

        last_w = self.params['W' + str(self.num_conv_layers + self.num_affine_layers-1)]
        last_b = self.params['b' + str(self.num_conv_layers + self.num_affine_layers-1)]
        scores, score_cache = affine_forward(next_input, last_w, last_b)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
          return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        affine_dx, affine_dw, affine_db = affine_backward(dscores, score_cache)
        grads['W'+str(self.num_conv_layers + self.num_affine_layers-1)] = affine_dw + self.reg * self.params['W'+str(self.num_conv_layers + self.num_affine_layers-1)]
        grads['b'+str(self.num_conv_layers + self.num_affine_layers-1)] = affine_db
        reg_loss = 0

        for w in [self.params[f] for f in self.params.keys() if f[0] == 'W']:
            reg_loss += 0.5 * self.reg * np.sum(w * w)

        loss += reg_loss

        for i in range(self.num_conv_layers + self.num_affine_layers-2, self.num_conv_layers-1, -1):
            affine_dx, affine_dw, affine_db, dgamma, dbeta = affine_bn_relu_backward(affine_dx, affine_cache[i]) 
            grads['beta'+str(i)]=dbeta
            grads['gamma'+str(i)]=dgamma
            grads['W' + str(i)] = affine_dw + self.reg * self.params['W' + str(i)]
            grads['b' + str(i)] = affine_db

        next_input = affine_dx
        for i in range(self.num_conv_layers-1, -1, -1):
            next_input, dw, db, dgamma, dbeta = conv_bn_relu_pool_backward(next_input, conv_cache[i])
            grads['beta'+str(i)] = dbeta
            grads['gamma'+str(i)] = dgamma
            grads['W' + str(i)] = dw + self.reg * self.params['W' + str(i)]
            grads['b' + str(i)] = db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, use_batchnorm=False,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.normal(0, weight_scale, (num_filters * H/2*W/2, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)
        
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'
    if self.use_batchnorm:
            for key, bn_param in self.bn_params.iteritems():
                bn_param[mode] = mode
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv_out, conv_cache = conv_forward_im2col(X, W1, b1, conv_param)
    conv_relu_out, conv_relu_cache = relu_forward(conv_out)
    pool_out, pool_cache = max_pool_forward_fast(conv_relu_out, pool_param)
    affine_relu_out, affine_relu_cache = affine_relu_forward(pool_out, W2, b2)
    affine_out, affine_cache = affine_forward(affine_relu_out, W3, b3)
    scores = affine_out
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg*(np.sum(self.params['W1']* self.params['W1']) + np.sum(self.params['W2']* self.params['W2'])+np.sum(self.params['W3']* self.params['W3']))
    affine_dx3, affine_dw3, affine_db3 = affine_backward(dscores, affine_cache)
    grads['W3'] = affine_dw3 + self.reg * self.params['W3']
    grads['b3'] = affine_db3
    affine_dx2, affine_dw2, affine_db2 = affine_relu_backward(affine_dx3, affine_relu_cache)
    grads['W2'] = affine_dw2 + self.reg * self.params['W2']
    grads['b2'] = affine_db2
    pool_dx = max_pool_backward_fast(affine_dx2, pool_cache)
    relu_dx = relu_backward(pool_dx, conv_relu_cache)
    dx, dw, db = conv_backward_im2col(relu_dx, conv_cache)
    grads['W1'] = dw + self.reg * self.params['W1']
    grads['b1'] = db
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
