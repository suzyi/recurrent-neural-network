from keras.layers import Activation, dot, concatenate
from keras.layers import Input, Dense, GRU
from keras.models import Model
from matplotlib import pyplot as plt
from matplotlib import cm # 3D surface color
from mpl_toolkits.mplot3d import Axes3D # projection='3d'
from time import time
import scipy.io as sio
import numpy as np


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# In[3]:


import numpy as np
import tensorflow as tf
import random as rn
from keras import backend as K

sd = 1
num_iter = 13
batch_sz = 122
time_begin = time()


np.random.seed(sd)
rn.seed(sd)
os.environ['PYTHONHASHSEED']=str(sd)


batch_sz = 122
activate_units = 109
config = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
tf.set_random_seed(sd)
sess = tf.Session(graph=tf.get_default_graph(), config=config)
K.set_session(sess)

# -----------------------------------------------------------------------------
dec_steps = 6
enc_steps = 14


kdv = sio.loadmat('twoSolitonKdV.mat')
data, x, t = kdv['u'], kdv['x'][0], kdv['t'][0]
q = 8
l = 5
Q = len(data[0])
Q = Q - Q%q
g = Q/q
in_dec_dim, out_dec_dim = q, q


# In[8]:


num_discard, num_train = 0, 88
num_test = 80-80%dec_steps
data, x, t = data[num_discard:, :Q], x[:Q], t[num_discard:]



latent_dim = 128
in_enc_dim = q+2*l


# In[11]:


enc_in_data = np.empty((0, enc_steps, in_enc_dim))
dec_in_data = np.empty((0, dec_steps, in_dec_dim))
dec_out_data = np.empty((0, dec_steps, out_dec_dim))

for k in range(g):
    if k==0:
        ind_start, ind_end = 0, q+2*l
    elif k==g-1:
        ind_start, ind_end = Q-q-2*l, Q
    else:
        ind_start, ind_end = k*q-l, (k+1)*q+l

    for i in range(num_train):
        enc_in_data = np.vstack(
            (enc_in_data, 
             data[i:i+enc_steps, ind_start:ind_end]
             .reshape(1, enc_steps, -1)
            ))
        dec_in_data = np.vstack(
            (dec_in_data, 
             data[i+enc_steps-1:i+enc_steps-1+dec_steps, k*q:(k+1)*q]
             .reshape(1, dec_steps, -1)
            ))
        dec_out_data = np.vstack(
            (dec_out_data, 
             data[i+enc_steps:i+enc_steps+dec_steps, k*q:(k+1)*q]
             .reshape(1, dec_steps, -1)
            ))
print(enc_in_data.shape)
# -----------------------------------------------------------------------------



i = num_train
data_start = data[i:i+enc_steps, :]
data_test = data[i+enc_steps:i+enc_steps+num_test, :]


# ## 2 - Build the Attention-based Seq2seq Model

# In[13]:


# build and train the model on the data collected above
# encoder
enc_in = Input(shape=(None, in_enc_dim), name='enc_in')
enc_GRU = GRU(units=latent_dim,
                return_sequences=True, 
                return_state=True, 
                name='enc_GRU')
enc_full_h, enc_h = enc_GRU(enc_in)

# decoder
dec_in = Input(shape=(None, in_dec_dim), name='dec_in')
dec_GRU = GRU(latent_dim, 
                return_sequences=True, 
                return_state=True,
                name='dec_GRU')
dec_full_h, _ = dec_GRU(dec_in, initial_state=enc_h)



scores = dot([dec_full_h, enc_full_h], axes=[2, 2])
weights = Activation('softmax', name='weights')(scores)

context = dot([weights, enc_full_h], axes=[2,1])
combined_context = concatenate([context, dec_full_h]) # quite resembles np.hstack((a, b))
activate_out = Dense(units=activate_units, activation="relu")(combined_context)
dense_out = Dense(units=out_dec_dim, activation="linear")(activate_out)


# In[16]:


# loss_type = 'mean_absolute_error'
loss_type = 'mean_squared_error'

opt_type = 'adam'
# opt_type = 'sgd'


# model
model = Model([enc_in, dec_in], dense_out)
model.compile(optimizer=opt_type, loss=loss_type)
model_history = model.fit([enc_in_data, dec_in_data], dec_out_data,
                                batch_size=batch_sz,
                                epochs=num_iter,
                                verbose=0,
                                validation_split=0.1)




def predictor(input_seq): # input_seq is a tensor of shape (1, enc_steps, dec_in_dim)
    input_seq = input_seq.reshape(1, enc_steps, -1)
    trigger = input_seq[0, -1, l:l+q]

    for i in range(dec_steps):
        # i=0, trigger.shape=(1, in_dec_dim)
        # i=1, trigger.shape=(2, in_dec_dim)
        # ...
        # i=dec_steps-1, trigger.shape=(dec_steps, in_dec_dim)
        outputs = model.predict([input_seq, trigger.reshape(-1, i+1, in_dec_dim)])
        trigger = np.vstack((trigger, outputs[:1, -1, :]))

    return outputs.reshape(dec_steps, -1)


# In[19]:


prediction = np.empty((0, Q))
for i in range(num_test/dec_steps):
    pred = np.empty((dec_steps, 0))
    for k in range(g):
        if k==0:
            ind_start, ind_end = 0, q+2*l
        elif k==g-1:
            ind_start, ind_end = Q-q-2*l, Q
        else:
            ind_start, ind_end = k*q-l, (k+1)*q+l
        pred = np.hstack((pred, predictor(data_start[:, ind_start:ind_end])))
    prediction = np.vstack((prediction, pred))

    data_start = np.vstack((data_start, pred))[-enc_steps:, :]

predict_error = np.square(np.subtract(data_test, prediction)).mean()



# ---------------------------------------------------------------
error = abs(data_test - prediction)
max_val = max([data_test.max(), prediction.max(), error.max()])
min_val = min([data_test.min(), prediction.min(), error.min()])
T, X = np.meshgrid(t, x)

fig, [ax_1, ax_2, ax_3] = plt.subplots(nrows=3, ncols=1, figsize=(2, 9))
ax_1.set_ylabel('x')
ax_2.set_ylabel('x')
ax_3.set_ylabel('x')
ax_3.set_xlabel('t')

mesh_1 = ax_1.pcolormesh(data_test.T[::-1], 
                         vmin=min_val, vmax=max_val, cmap = cm.coolwarm)
mesh_2 = ax_2.pcolormesh(prediction.T[::-1], 
                         vmin=min_val, vmax=max_val, cmap = cm.coolwarm)
mesh_3 = ax_3.pcolormesh(error.T[::-1], 
                         vmin=min_val, vmax=max_val, cmap = cm.coolwarm)

fig.colorbar(mesh_1, ax=[ax_1, ax_2, ax_3], shrink=.5)
fig.show()
# ---------------------------------------------------------------
# 1, 30, 50, 70 are very important, 
# up to now, 30 is achieved.
figure, axises = plt.subplots(nrows=3, ncols=1, figsize=(8, 9))
figure.suptitle('Original states of Lorenz system')
for step, axis in zip([0, 30, 50], axises):
    axis.plot(x, data_test[step, :], 'k-', x, prediction[step, :], 'b--')
    axis.set_ylabel('u')
    axis.legend(["actual", "predicted"], loc='upper left', ncol=1)
axis.set_xlabel('x')
plt.show()
# ---------------------------------------------------------------


time_cost = (time()-time_begin)/60.0
print(predict_error)

# In[24]:


#     List = [predict_error, enc_steps, dec_steps, latent_dim, num_discard, num_train, num_test, batch_sz, num_iter, time_cost, activate_units, q, l, loss_type, opt_type, sd]
#     Name = 'predict_error, enc_steps, dec_steps, latent_dim, num_discard, num_train, num_test, batch_sz, num_iter, time_cost, activate_units, q, l, loss_type, opt_type, sd \n'
#     f = open('KdV-seq2seq-training-process.csv','a')
# #     f.write(Name)
#     for i in List:
#         f.write('{},'.format(str(i)))
#     f.write('\n')
#     f.close()
