import cudamat as cm
import numpy as np
import time
from lstm_layer import *
from lstm_layer_cuda import *
import pdb
import cPickle as cp
cm.cuda_set_device(0)
cm.init()


#------------------------------LSTM forward CPU version-------------------------------#
lstm_layer_obj  = Layer()
lstm_layer_obj_cuda = Layer_cuda()
input_size = 5
hidden_size = 10
num_classes = 10
X = np.random.randn(input_size,input_size)
X_frwd  = np.zeros((X.shape[0]+1,X.shape[1]))
X_frwd[1:] = X
X_frwd=X_frwd.astype(np.float32)
h = np.zeros((X_frwd.shape[0],hidden_size))
h=h.astype(np.float32)
print '\n'

model={}
model['W_xi'] = np.random.randn(input_size,hidden_size).astype(np.float32)
model['W_xf'] = np.random.randn(input_size,hidden_size).astype(np.float32)
model['W_xo'] = np.random.randn(input_size,hidden_size).astype(np.float32)
model['W_xg'] = np.random.randn(input_size,hidden_size).astype(np.float32)
model['W_hi'] = np.random.randn(hidden_size,hidden_size).astype(np.float32)
model['W_hf'] = np.random.randn(hidden_size,hidden_size).astype(np.float32)
model['W_ho'] = np.random.randn(hidden_size,hidden_size).astype(np.float32)
model['W_hg'] = np.random.randn(hidden_size,hidden_size).astype(np.float32)
model['b_i'] = np.random.randn(1,hidden_size).astype(np.float32)
model['b_f'] = np.random.randn(1,hidden_size).astype(np.float32)
model['b_o'] = np.random.randn(1,hidden_size).astype(np.float32)
model['b_g'] = np.random.randn(1,hidden_size).astype(np.float32)


start = time.time()
out,cache = lstm_layer_obj.forward_propagation(X_frwd,model,h)
end = time.time()
print 'Total Time Forward norm =%f'% (end-start)

print 'Backpropagation !!!!\n'
dout=np.random.randn(out.shape[0],out.shape[1]) 
start = time.time()
dx,grads = lstm_layer_obj.backward_propagation(dout,cache)
end = time.time()
print 'Total Time backward norm =%f \n'% (end-start)

print out
print '\n'
print dx
print 'Now cuda \n'

#---------------------------------------GPU version using CUDAMAT----------------------------# 
X_frwd_cuda = np.zeros((X.shape[0],X.shape[1]+1))
X_frwd_cuda[:,1:]=X.T
X_frwd_cuda= X_frwd_cuda.astype(np.float32)
h_cuda = np.zeros((hidden_size,X_frwd_cuda.shape[1]))
h_cuda = h_cuda.astype(np.float32)
model['W_xi'] = cm.CUDAMatrix(model['W_xi'].T) 
model['W_xf'] = cm.CUDAMatrix(model['W_xf'] .T)
model['W_xo'] = cm.CUDAMatrix(model['W_xo'].T)
model['W_xg'] = cm.CUDAMatrix(model['W_xg'].T)
model['W_hi'] = cm.CUDAMatrix(model['W_hi'].T)
model['W_hf'] = cm.CUDAMatrix(model['W_hf'].T)
model['W_ho'] = cm.CUDAMatrix(model['W_ho'].T)
model['W_hg'] = cm.CUDAMatrix(model['W_hg'].T)
model['b_i'] = cm.CUDAMatrix(model['b_i'].T)
model['b_f'] = cm.CUDAMatrix(model['b_f'].T)
model['b_o'] = cm.CUDAMatrix(model['b_o'].T)
model['b_g'] = cm.CUDAMatrix(model['b_g'].T)

#pdb.set_trace()
start=time.time()
out,cache_cuda = lstm_layer_obj_cuda.forward_propagation_restructure(X_frwd_cuda,model,h_cuda)
end = time.time()
print 'Total time forward cuda =%f\n'%(end-start)

out_lstm=out.transpose()
print out_lstm.asarray()
print '\n'
dout_cuda = cm.CUDAMatrix(dout.T) 
start = time.time()
dx_cuda,grads_cuda = lstm_layer_obj_cuda.backward_propagation_restructure(dout_cuda,cache_cuda)
end = time.time()
print 'Total Time backprop cuda =%f\n' % (end-start)
dx_cuda= dx_cuda.transpose()
print dx_cuda.asarray()

cm.shutdown()


