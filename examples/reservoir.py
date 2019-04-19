"""
Reservoir computing for prediction purpose.
Written by Guorui Shen, guorui233@outlook.com
Apr 17~18, 2019.

References:
    [1] Jaeger H, Haas H. Harnessing nonlinearity: predicting chaotic systems and saving energy in wireless communication[J]. Science, 2004, 304(5667):78-80.
    [2] Pathak J, Hunt B, Girvan M, et al. Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach[J]. Phys.rev.lett, 2018, 120(2).
    [3] https://mantas.info/code/simple_esn/

unsolved questions:
    2. Annotation
    
How to use?
    step 1(import). Make sure include "from reservoir import Reservoir" in your main function
    step 2(build model). model = Reservoir(dim_in, dim_r, dim_out, alpha=.3, spectral_radius=1.25)
    step 3(train model). Lorenz_reservoir.train(features, labels, num_init, lam)
    step 4(make prediction). next_state = Lorenz_reservoir.predict(u)
"""

import numpy as np

class Reservoir(object):
    """
    Build a reservoir and related internal functions.
    This Reservoir is designed particularly for prediction purpose.
    
    Parameter:
        dim_in: a scalar, dimension of input.
        dim_r: a scalar, dimension of reservoir.
        dim_out: a scalar, dimension of output.
        alpha: a scalar.
    """
    
    def __init__(self, dim_in, dim_r, dim_out, alpha=.3, spectral_radius=1.25):
        
        # dimension of network
        self._dim_in = dim_in
        self._dim_reservoir = dim_r
        self._dim_out = dim_out
        
        # attributes of reservoir
        self._alpha = alpha
        self._spectral_radius = spectral_radius
        self._reservoir_state = np.zeros((1, dim_r))

        # Initialize weights of input, reservoir and output layer.
        self._input_weight = np.random.rand(dim_in+1, dim_r)-.5,  # dim_in+1 where 1 is for bias
        self._reservoir_weight = np.random.rand(dim_r, dim_r)-.5
        self._output_weight = np.zeros((1+dim_in+dim_r, dim_out)) # 1+dim_in+dim_r is for bias+input+reservoir        

    def train(self, features, labels, num_init, lam):
        """
        During the training process, reservoir weight (with biases incorporated in), output weight (with biases incorporated in) as well as reservoir state will be determined.
        
        input:
            features: each row is an example.
            labels: each row is a label for corresponding feature.
        return: None
        """
        alpha = self._alpha
        dim_in, dim_r, spectral_radius = self._dim_in, self._dim_reservoir, self._spectral_radius

        # compute reservoir weight
        Wr = self._reservoir_weight
        max_eigvalue = max(abs(np.linalg.eig(Wr)[0]))
        Wr *= spectral_radius / max_eigvalue
        self._reservoir_weight = Wr

        # compute output weight
        len_train = len(features)
        R = np.zeros((len_train-num_init, 1+dim_in+dim_r))
        r = self._reservoir_state
        
        for t in range(len_train):
            u = features[t]  # t-th feature
            r = (1-alpha)*r + alpha*np.tanh(np.matmul(r, self._reservoir_weight) + \
                                            np.matmul(np.hstack((1, u)), self._input_weight)) # bias was incorporated into input weight
            if t>=num_init:
                R[t-num_init, :] = np.hstack((1,u,r[0]))
                
        self._output_weight = np.matmul(np.linalg.inv(np.matmul(R.T, R) + lam*np.eye(1+dim_in+dim_r)),\
                                        np.matmul(R.T, labels[num_init:,:]))
        
        # update reservoir_state
        self._reservoir_state = r # at index=len_train
    
    def predict(self, u):
        """
        This predictor is intended to predict one step ahead, given input "u".
        
        input:
            u: current input of system
        
        return:
            state: one step ahead prediction.
        """
        dim_out = self._dim_out
        alpha = self._alpha
        r = self._reservoir_state         # current reservoir state

        bias_and_u = np.hstack((1, u))
        r = (1-alpha)*r + alpha*np.tanh(np.matmul(r, self._reservoir_weight) + np.matmul(bias_and_u, self._input_weight))
        bias_input_r = np.hstack((1, u, r[0]))
        state = np.matmul(bias_input_r, self._output_weight)
        
        # update reservoir state
        self._reservoir_state = r
        
        return state