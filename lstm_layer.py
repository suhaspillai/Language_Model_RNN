import numpy as np
import pdb

class Layer:
    '''
    Layers of deep neural network  
    '''
    
    def __init__(self):
        pass
    
    def affine_forward(self,x, w, b):

        """
        Computes the forward pass for an affine (fully-connected) layer.

        The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
        We multiply this against a weight matrix of shape (D, M) where
        D = \prod_i d_i

        Inputs:
        x - Input data, of shape (N, d_1, ..., d_k)
        w - Weights, of shape (D, M)
        b - Biases, of shape (M,)

        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        out = None
        x_new=x.reshape(x.shape[0],np.prod(x.shape[1:]))
        out=x_new.dot(w)+b.T    #change to match dimensions
        cache = (x, w, b)
        return out, cache


    def affine_backward(self,dout, cache):
        """
        Computes the backward pass for an affine layer.

        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
        - x: Input data, of shape (N, d_1, ... d_k)
        - w: Weights, of shape (D, M)

        Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache
        x_new=x.reshape(x.shape[0],np.prod(x.shape[1:]))
        dx, dw, db = None, None, None
        dw=np.zeros(w.shape)
        db=np.zeros(b.shape)
        dx=np.zeros(x_new.shape)
        N =x.shape[0]
        db = db + np.sum(dout,axis=0)
        dx=dout.dot(w.T)
        dx=dx.reshape(x.shape)
        dw=(x_new.T).dot(dout)
        return dx, dw, db

    def tanh_forward(self,x):

        """
        Forwrad pass with tanh activation

        Input:
        x - input to tanh layer

        Output :
        out - output after tanh activation
        cache - (x)
        """       
        out = None
        out = np.tanh(x)
        cache = (x)
        return out,cache


    def tanh_backward(self,dout,cache):
        '''
        Backward pass for tanh layer.

        Input:
        dout -  Gradients from top layer.
        cache - (x)

        Output:
        dx -  gradient with respect to input
        '''
        x = cache
        temp = np.tanh(x)
        dx = dout * (1-(temp**2))
        return dx

    def sigmoid(self, X):
        ''' 
        Sigmoid activation 
        '''
        x_exp = np.exp(-X)
        x_exp = x_exp.astype(np.float32)   
        x_sigmoid = 1/(1+x_exp) 
        x_sigmoid=x_sigmoid.astype(np.float32)
        return x_sigmoid

    def tanh(self,X):
        val = np.tanh(X)
        '''
        tanh activation
        '''
        val = val.astype(np.float32)  
        return val

    def forward_propagation(self, X, model,h):
        '''
        Naive forward pass for lstm layer.

        Input:
        X - Input to LSTM layer. T * N , where T - number of time steps, N - number of input features
        model - Weights for hidden layer, for LSTM block with one cell.
        h - hidden layer for lstm layer, matrix of all zeros. Dimensions same as X

        Output:
        h - hidden layer with LSTM block. T*N ,where N is number of hidden units,
        T is number of time steps.
        cache - (X,model,h,cell_state,arr_i,arr_f,arr_o,arr_g)
        '''
               
        T = X.shape[0]
        W_xi = model['W_xi']
        W_xf = model['W_xf']
        W_xo = model['W_xo'] 
        W_xg = model['W_xg']
        W_hi = model['W_hi']
        W_hf = model['W_hf']
        W_ho = model['W_ho']
        W_hg = model['W_hg']
        b_i = model['b_i']
        b_f = model['b_f'] 
        b_o = model['b_o']
        b_g = model['b_g']

        # Initialization for temporary storage
        cols = W_xi.shape[1]
        arr_i = np.zeros((T,cols)).astype(np.float32)
        arr_f = np.zeros((T,cols)).astype(np.float32)
        arr_o = np.zeros((T,cols)).astype(np.float32)
        arr_g = np.zeros((T,cols)).astype(np.float32) 
        cell_state = np.zeros((T,cols)).astype(np.float32)

        
        for t in xrange(1,T):
            arr_i[t] = np.dot(X[t],W_xi) + np.dot(h[t-1],W_hi) + b_i
            arr_f[t] = np.dot(X[t],W_xf) + np.dot(h[t-1],W_hf) + b_f
            arr_o[t] = np.dot(X[t],W_xo) + np.dot(h[t-1],W_ho) + b_o
            arr_g[t] = np.dot(X[t],W_xg) + np.dot(h[t-1],W_hg) + b_g
            i_gate = self.sigmoid(arr_i[t]) # output from input gate 
            f_gate = self.sigmoid(arr_f[t]) # output from forget gate
            o_gate = self.sigmoid(arr_o[t]) # output from output gate 
            g_input = self.tanh(arr_g[t])   # output from input
            # calculating cell state
            cell_state[t] = i_gate * g_input  +  f_gate * cell_state[t-1]
            # output from LSTM block
            h[t] = o_gate * self.tanh(cell_state[t])
            
        cache = (X,model,h,cell_state,arr_i,arr_f,arr_o,arr_g)
        return h,cache     



    def backward_propagation(self,dout,cache):
        '''
        Backward pass for LSTM layer.
        Input :
        dout - gradients from top layer
        cache - (X,model,h,cell_state,arr_i,arr_f,arr_o,arr_g)

        Output:
        dx - Gradients with respect to input
        grads - Gradients with respect to weights and bias of LSTM block with one cell and 3 gates
        '''
        
        T = dout.shape[0] 
        X,model,h,cell_state,arr_i,arr_f,arr_o,arr_g = cache
        W_xi = model['W_xi']
        W_xf = model['W_xf']
        W_xo = model['W_xo']
        W_xg = model['W_xg']
        W_hi = model['W_hi']
        W_hf = model['W_hf']
        W_ho = model['W_ho']
        W_hg = model['W_hg']
        b_i = model['b_i']
        b_f = model['b_f']
        b_g = model['b_g']
        b_o = model['b_o']
        cols = dout.shape[1] 
        n_gates = 4
        dout_ec = np.zeros((T,cols))  
        dout_h = np.zeros((T+1,n_gates,cols)) # to keep derivatives of gates
        dout_es = np.zeros((T+1,cols))  # to keep cell_state derivative
        arr_f_back_pass = np.zeros((arr_f.shape[0]+1,arr_f.shape[1]))
        arr_f_back_pass[:-1] = arr_f

        dW_xi = np.zeros(W_xi.shape)
        dW_xf = np.zeros(W_xf.shape)
        dW_xo = np.zeros(W_xo.shape)
        dW_xg = np.zeros(W_xg.shape)
        dW_hi = np.zeros(W_hi.shape)
        dW_hf = np.zeros(W_hf.shape)
        dW_ho = np.zeros(W_ho.shape)
        dW_hg = np.zeros(W_hg.shape)
        db_i = np.zeros(b_i.shape)
        db_f = np.zeros(b_f.shape)
        db_o = np.zeros(b_o.shape)
        db_g = np.zeros(b_g.shape)
        dx = np.zeros(X.shape)
        N_x = X[0].shape[0]
        N_h = h[0].shape[0]


        for t in reversed(xrange(1,T)):
            dout_ec[t] = dout[t] + np.dot(W_hi,dout_h[t+1,0]) + np.dot(W_hf,dout_h[t+1,1]) + np.dot(W_ho,dout_h[t+1,2]) + np.dot(W_hg,dout_h[t+1,3])
            dout_h[t,2] = (dout_ec[t] * self.tanh(cell_state[t])) * self.sigmoid(arr_o[t]) * (1-self.sigmoid(arr_o[t]))
            dout_es[t] = self.sigmoid(arr_o[t]) * (1-(self.tanh(cell_state[t]))**2) * dout_ec[t] + (dout_es[t+1] * self.sigmoid(arr_f_back_pass[t+1]))
            dout_h[t,1] = dout_es[t] * cell_state[t-1] * (self.sigmoid(arr_f_back_pass[t])*(1-self.sigmoid(arr_f_back_pass[t])))
            dout_h[t,0] = dout_es[t] * self.tanh(arr_g[t]) * (self.sigmoid(arr_i[t])*(1-self.sigmoid(arr_i[t])))
            dout_h[t,3] = dout_es[t] * self.sigmoid(arr_i[t])  * (1-(self.tanh(arr_g[t]))**2)

            # Calculating the gradients
            temp_x =  X[t].reshape(N_x,1)
            temp_h = h[t-1].reshape(N_h,1)
            dW_xi = dW_xi + temp_x * dout_h[t,0]
            dW_xf = dW_xf + temp_x * dout_h[t,1]
            dW_xo = dW_xo + temp_x * dout_h[t,2]
            dW_xg = dW_xg + temp_x * dout_h[t,3]
            dW_hi = dW_hi + temp_h * dout_h[t,0]
            dW_hf = dW_hf + temp_h * dout_h[t,1]
            dW_ho = dW_ho + temp_h * dout_h[t,2]
            dW_hg = dW_hg + temp_h * dout_h[t,3]
            db_i = db_i + dout_h[t,0]
            db_f = db_f + dout_h[t,1]
            db_o = db_o + dout_h[t,2]
            db_g = db_g + dout_h[t,3]
            dx[t] = np.dot(W_xi,dout_h[t,0]) + np.dot(W_xf,dout_h[t,1]) + np.dot(W_xo,dout_h[t,2])+ np.dot(W_xg,dout_h[t,3]) 

        grads ={'W_xi':dW_xi,'W_xf':dW_xf,'W_xo':dW_xo, 'W_xg':dW_xg,'W_hi':dW_hi, 'W_hf':dW_hf, 'W_ho':dW_ho, 'W_hg':dW_hg, 'b_i':db_i,'b_f':db_f,'b_o':db_o,'b_g':db_g}
        return dx, grads



    def softmax_loss(self,x, y):
        """
        Computes the loss and gradient for softmax classification.

        Inputs:
        - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
          for the ith input.
        - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
          0 <= y[i] < C

        Returns a tuple of:
        - loss: Scalar giving the loss
        - dx: Gradient of the loss with respect to x
        """
        #pdb.set_trace()
        f_open = open('prob.txt','a')
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        #print ('\n softmax probabilities \n')

        probs /= np.sum(probs, axis=1, keepdims=True)
        f_open.write(str(probs))

        N = x.shape[0]

        loss = -np.sum(np.log(probs[np.arange(N), y])) / N
        #temp = np.log(probs[np.arange(N), y])
        #loss = -np.sum(temp[1:]) / N-1

        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        #dx[0] = 0    # The sequence starts from 1 , 0 is just a dummy entry
        return loss, dx

