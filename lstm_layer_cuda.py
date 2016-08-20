import numpy as  np
import cudamat as cm
import pdb

class Layer_cuda():
    '''
    Class contains layers for building deep neural nets 
    '''
    
    def __init__(self):
        pass


    def sigmoid(self, X):
        '''
        Performs sigmoid activation 
        ''' 
        row,col= X.shape
        x_exp = cm.empty((row,col))
        X.mult(-1)  
        cm.exp(X, target = x_exp)
        x_exp.add(1)
        X_ones = cm.CUDAMatrix(np.ones((row,col)))        
        X_ones.divide(x_exp)
        #x_sigmoid = 1/(1+x_exp) 
        return X_ones


    def tanh(self,X):
        '''
        Perform tanh activation  
        ''' 
        row,col= X.shape
        X_ones = cm.CUDAMatrix(np.ones((row,col)))
        x_exp = cm.empty((row,col))
        x_exp.assign(0)
        X.mult(2)
        cm.exp(X, target = x_exp)
        X_sub = cm.CUDAMatrix(np.ones((row,col)))
        X_add = cm.CUDAMatrix(np.ones((row,col)))
        x_exp.subtract(X_ones,target = X_sub)
        x_exp.add(X_ones,target = X_add)
        X_sub.divide(X_add)
        return X_sub

    def affine_forward_cuda(self,x, w, b):

        """
        Computes the forward pass for an affine (fully-connected) layer.
        Inputs:
        x - Input data, of shape (N, d_1, ..., d_k)
        w - Weights, of shape (D, M)
        b - Biases, of shape (M,)

        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """

        out = None
        cm.dot(w,x,out)
        out.add(b)
        cache = (x, w, b)
        return out, cache

    def affine_backward_cuda(self,dout, cache):
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
        dx, dw, db = None, None, None
        dw = cm.empty(w.shape)
        dw.assign(0)
        db = cm.empty(b.shape)
        db.assign(0)
        dx=np.zeros(x_new.shape)
        dx = cm.empty(x.shape)
        db = db + np.sum(dout,axis=0)
        dout.sum(axis = 1, target = db) 
        cm.dot(w.T,dout,dx)
        cm.out(X,dout.T,dw)   
        return dx, dw, db
       
   

    def forward_propagation_restructure(self, X, model,h):
        '''
        The method computes forward propagfation for LSTM

        Input :
        X - 1D Input to the lstm layer,the first column is padded with zeros. size N * T,where N is number of inputs,
        T is number of time steps. 
        model - Weight for LSTM layer
        h - hidden layer for lstm layer, matrix of all zeros. Dimensions same as X
        

        output:
        h - hidden layer for lstm layer, matrix of all zeros. N * T ,where N is number of hidden units,
        T is number of time steps.
        cache - (X,model,h,cell_state,arr_i,arr_f,arr_o,arr_g)
        '''
        
        X = cm.CUDAMatrix(X)
        h = cm.CUDAMatrix(h)
        T = X.shape[1]
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
        
        cols = W_xi.shape[0]
        # Stores input gate values
        arr_i = cm.empty((cols,T))
        arr_i.assign(0)
        # Stores forget values
        arr_f = cm.empty((cols,T))
        arr_f.assign(0)
        # Stores output gate values.
        arr_o = cm.empty((cols,T))
        arr_o.assign(0)
        # Store input values 
        arr_g = cm.empty((cols,T))
        arr_g.assign(0)
        # For cell state for every cell in lstm.
        cell_state = cm.empty((cols,T))
        cell_state.assign(0)
        
        temp_x = cm.empty((X.shape[0],1))
        temp_x.assign(0)
        temp_h = cm.empty((h.shape[0],1)) 
        temp_h.assign(0) 
        temp_i = cm.empty((arr_i.shape[0],1))
        temp_f = cm.empty((arr_i.shape[0],1))
        temp_o = cm.empty((arr_i.shape[0],1))
        temp_g = cm.empty((arr_i.shape[0],1))
        
        
        for t in xrange(1,T):
            
            temp_x.assign(X.get_col_slice(t,t+1))
            temp_h.assign(h.get_col_slice(t-1,t))
            temp_i.assign(0)

            # Calculation for input gate 
            temp_i.add(cm.dot(W_xi,temp_x)) 
            temp_i.add(cm.dot(W_hi,temp_h))
            temp_i.add(b_i)     # no of rows in weight
            arr_i.set_col_slice(t,t+1,temp_i)
            i_gate = cm.sigmoid(temp_i)

            # Calculation for forget gate
            temp_f.assign(0)
            temp_f.add(cm.dot(W_xf,temp_x))
            temp_f.add(cm.dot(W_hf,temp_h))
            temp_f.add(b_f)
            arr_f.set_col_slice(t,t+1,temp_f) 
            f_gate = cm.sigmoid(temp_f)

            #Calculation for output gate
            temp_o.assign(0)
            temp_o.add(cm.dot(W_xo,temp_x))
            temp_o.add(cm.dot(W_ho,temp_h))
            temp_o.add(b_o) 
            arr_o.set_col_slice(t,t+1,temp_o)
            o_gate = cm.sigmoid(temp_o)

            # Calculation for input.
            temp_g.assign(0)
            temp_g.add(cm.dot(W_xg,temp_x))
            temp_g.add(cm.dot(W_hg,temp_h))
            temp_g.add(b_g)
            arr_g.set_col_slice(t,t+1,temp_g)
            g_input = cm.tanh(temp_g)            

            i_gate.mult(g_input)
            f_gate.mult(cell_state.get_col_slice(t-1,t))
            i_gate.add(f_gate)
            cell_state.set_col_slice(t,t+1,i_gate)
            o_gate.mult(cm.tanh(i_gate))
            h.set_col_slice(t,t+1,o_gate)

        cache = (X,model,h,cell_state,arr_i,arr_f,arr_o,arr_g)
        # Need to delete aal the temproray matrces created, for some reason these are not garbage collected.
        # when using CUDAMAT
        
        del temp_i,temp_f,temp_o,temp_g,temp_x,temp_h       
        return h,cache

        

    def backward_propagation_restructure(self,dout,cache):
        '''
        Backward pass for LSTM layer

        Input:
        dout - error derivatives from the top layers
        cache - (X,model,h,cell_state,arr_i,arr_f,arr_o,arr_g)

        Output:
        dx - gradients with respect to input
        grads - dictionary containing gradients with respect to all the gates 
        '''

        T = dout.shape[1] 
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
        cols = dout.shape[0] 
        n_gates = 4   # for input,forget,output gate and one more for Input 
        dout_ec = cm.empty((cols,T))
        dout_ec.assign(0)
        #--------------for all 4 gates------------------------
        dout_h_0 = cm.empty((cols,T+1))
        dout_h_0.assign(0)
        dout_h_1= cm.empty((cols,T+1))
        dout_h_1.assign(0)
        dout_h_2= cm.empty((cols,T+1))
        dout_h_2.assign(0)
        dout_h_3= cm.empty((cols,T+1))
        dout_h_3.assign(0)
        dout_es = cm.empty((cols,T+1))
        dout_es.assign(0)
        arr_f_back_pass = cm.empty((arr_f.shape[0],arr_f.shape[1]+1))
        arr_f_back_pass.set_col_slice(0,arr_f_back_pass.shape[1]-1,arr_f)

        # we can assign memory beforehand 
        dW_xi = cm.CUDAMatrix(np.zeros(W_xi.shape))
        dW_xf = cm.CUDAMatrix(np.zeros(W_xf.shape))
        dW_xo = cm.CUDAMatrix(np.zeros(W_xo.shape))
        dW_xg = cm.CUDAMatrix(np.zeros(W_xg.shape))
        dW_hi = cm.CUDAMatrix(np.zeros(W_hi.shape))
        dW_hf = cm.CUDAMatrix(np.zeros(W_hf.shape))
        dW_ho = cm.CUDAMatrix(np.zeros(W_ho.shape))
        dW_hg = cm.CUDAMatrix(np.zeros(W_hg.shape))
        db_i = cm.CUDAMatrix(np.zeros(b_i.shape))
        db_f = cm.CUDAMatrix(np.zeros(b_f.shape))
        db_o = cm.CUDAMatrix(np.zeros(b_o.shape))
        db_g = cm.CUDAMatrix(np.zeros(b_g.shape))
        dx = cm.CUDAMatrix(np.zeros(X.shape))
        
        #Below variables are used to store intermediate values. 
        temp_ec = cm.empty((dout_ec.shape[0],1))
        temp_es = cm.empty(temp_ec.shape)
        temp_ones = cm.empty(temp_ec.shape)     
        temp = cm.empty(temp_ec.shape)
        temp_w = cm.empty(W_xi.shape)

        
        for t in reversed(xrange(1,T)):
            temp_ec.assign(0)
            temp_es.assign(0)
            temp.assign(0) 
            temp_ones.assign(1)
            
            temp_ec.add(dout.get_col_slice(t,t+1)) 
            temp_ec.add(cm.dot(W_hi.T,dout_h_0.get_col_slice(t+1,t+2)))
            temp_ec.add(cm.dot(W_hf.T,dout_h_1.get_col_slice(t+1,t+2))) 
            temp_ec.add(cm.dot(W_ho.T,dout_h_2.get_col_slice(t+1,t+2)))
            temp_ec.add(cm.dot(W_hg.T,dout_h_3.get_col_slice(t+1,t+2))) 
            dout_ec.set_col_slice(t,t+1,temp_ec)

            # dout_h_2  
            cm.tanh(cell_state.get_col_slice(t,t+1),temp) 
            temp_ec.mult(temp)
            cm.sigmoid(arr_o.get_col_slice(t,t+1),temp)
            temp_ec.mult(temp) 
            temp_ones.subtract(temp) 
            temp_ec.mult(temp_ones) 
            dout_h_2.set_col_slice(t,t+1,temp_ec)
           
            temp_ones.assign(1)
            temp_ec.assign(dout_ec.get_col_slice(t,t+1))
            cm.sigmoid(arr_o.get_col_slice(t,t+1),temp)
            temp_ec.mult(temp)
            cm.tanh(cell_state.get_col_slice(t,t+1),temp)
            cm.pow(temp,2)
            temp_ones.subtract(temp)
            temp_ec.mult(temp_ones)
            
            temp_es.assign(dout_es.get_col_slice(t+1,t+2))
            cm.sigmoid(arr_f_back_pass.get_col_slice(t+1,t+2),temp)
            temp_es.mult(temp)
            temp_es.add(temp_ec)
            dout_es.set_col_slice(t,t+1,temp_es)
            temp_ones.assign(1)
            temp_es.mult(cell_state.get_col_slice(t-1,t))
            cm.sigmoid(arr_f_back_pass.get_col_slice(t,t+1),temp)
            temp_es.mult(temp)
            temp_ones.subtract(temp)
            temp_es.mult(temp_ones)            

            #dout_h_1  
            dout_h_1.set_col_slice(t,t+1,temp_es)  
            temp_ones.assign(1)
            temp_es.assign(dout_es.get_col_slice(t,t+1))
            cm.tanh(arr_g.get_col_slice(t,t+1),temp)  
            temp_es.mult(temp)   
            cm.sigmoid(arr_i.get_col_slice(t,t+1),temp)
            temp_es.mult(temp)
            temp_ones.subtract(temp)
            temp_es.mult(temp_ones) 
            dout_h_0.set_col_slice(t,t+1,temp_es)
           

            # dout_h_3
            
            temp_ones.assign(1)
            temp_es.assign(dout_es.get_col_slice(t,t+1))
            cm.sigmoid(arr_i.get_col_slice(t,t+1),temp)
            temp_es.mult(temp)
            cm.tanh(arr_g.get_col_slice(t,t+1),temp)
            cm.pow(temp,2) 
            temp_ones.subtract(temp)
            temp_es.mult(temp_ones) 
            dout_h_3.set_col_slice(t,t+1,temp_es)
             

        # Gradients with respect to input
        dW_xi = cm.dot(X,dout_h_0.get_col_slice(0,T).T)

        # Gradient with respect to Weights
        dW_xf  = cm.dot(X,dout_h_1.get_col_slice(0,T).T) 
        dW_xo = cm.dot(X,dout_h_2.get_col_slice(0,T).T) 
        dW_xg = cm.dot(X,dout_h_3.get_col_slice(0,T).T)
        dW_hi = cm.dot(h.get_col_slice(0,T),dout_h_0.get_col_slice(1,T+1).T)
        dW_hf = cm.dot(h.get_col_slice(0,T),dout_h_1.get_col_slice(1,T+1).T)
        dW_ho = cm.dot(h.get_col_slice(0,T),dout_h_2.get_col_slice(1,T+1).T) 
        dW_hg = cm.dot(h.get_col_slice(0,T),dout_h_3.get_col_slice(1,T+1).T)
        
        # Gradients with respect to bias
        dout_h_0.sum(axis = 1, target = db_i)
        dout_h_1.sum(axis = 1, target = db_f)
        dout_h_2.sum(axis = 1, target = db_o)
        dout_h_3.sum(axis = 1, target = db_g)
        dx.add(cm.dot(W_xi.T,dout_h_0).get_col_slice(0,T))
        dx.add(cm.dot(W_xf.T,dout_h_1).get_col_slice(0,T))
        dx.add(cm.dot(W_xo.T,dout_h_2).get_col_slice(0,T))
        dx.add(cm.dot(W_xg.T,dout_h_3).get_col_slice(0,T))

        grads ={'W_xi':dW_xi,'W_xf':dW_xf,'W_xo':dW_xo, 'W_xg':dW_xg,'W_hi':dW_hi, 'W_hf':dW_hf, 'W_ho':dW_ho, 'W_hg':dW_hg, 'b_i':db_i,'b_f':db_f,'b_o':db_o,'b_g':db_g}

        # Need to delete all the temporary matrices,for some reason this is not garabage collected when I use CUDAMAT.Need to check 
        del temp_ec
        del temp_es
        del temp_ones
        del temp
        del temp_w
        del dout_h_0
        del dout_h_1
        del dout_h_2
        del dout_h_3
        del dout_ec
        del dout_es   
        del arr_f_back_pass
        del cache          
        return dx, grads 
   
 
                     
