# Recurrent Neural Network
May 13, 2019.
Guorui Shen, guorui233@outlook.com

## 1 - What is Recurrent Neural Network?
Recurrent Neural Networks (RNNs) is actually a first-order Hidden Markov Model (HMM) which has the form
```
h_t=f(x_t,h_{t-1};\theta),\\
y_t = g(h_t;\theta),
```
where $x_t, h_t, y_t$ denote the input, hidden state and output at time $t$, respectively. A common parametrization scheme for above equation is 

```
\begin{align}
h_t&=f(W_{hx}x_t+W_{hh}h_{t-1}+b_{h}),\cr
x_{t+1}&= W_{xh}h_t+b_{x}.
\end{align}
```
RNNs have many variants according to various RNN cells, including Vanilla RNN (or an Elman RNN after Prof. Jeffrey Elman, the most simple kind), Long Short Term Memory (LSTM, 1997), Bidirectional LSTM, Gated Recurrent Units (GRU, 2014), Encoder-Decoder sequence-to-sequence architecture, Reservoir Computing or Echo State Network (RC or ESN) and etc.

## 2 - Application of RNN
RNNs are mainly used in time series analysis, automatic image captioning, sentiment classification, machine traslation, chatbot and etc.

## 3 - How to Implement a RNN?
As for how to implement a RNN in real situation, here we have a list of hands-on examples, based on tensorflow.
### Tutorials
+ [tensorflow-seq2seq-tutorials](https://github.com/ematvey/tensorflow-seq2seq-tutorials).
### Basics
+ Vanilla unit - `tf.contrib.rnn.BasicRNNCell(hidden_dim)`
+ LSTM unit - `tf.contrib.rnn.BasicLSTMCell(hidden_dim)`, `tf.contrib.rnn.LSTMCell(hidden_dim)`, 
+ GRU - `tf.contrib.rnn.GRUCell(hidden_dim)`

### Tutorials and Applications
| Date | Codes | Data |
|---| -------- |-------- |
| Jun 30, 2019 | [Attention-based-Seq2seq-Eng2Jap-Keras.ipynb](https://github.com/suzyi/recurrent-neural-network/blob/master/notebooks/Attention-based-Seq2seq-Eng2Jap-Keras.ipynb) | [eng2jap.csv](https://github.com/suzyi/recurrent-neural-network/tree/master/data/eng2jap.csv) |
| Jun 24, 2019 | [English2French-keras-tf-seq2seq-tensorboard.ipynb](https://github.com/suzyi/recurrent-neural-network/blob/master/notebooks/English2French-keras-tf-seq2seq-tensorboard.ipynb) | [fra.txt](https://github.com/suzyi/recurrent-neural-network/tree/master/data/fra.txt) |
| Jun 16, 2019 | [seq2seq-vs-GPR-Lorenz-prediction-tensorflow-xyz2xyz.ipynb](https://github.com/suzyi/recurrent-neural-network/blob/master/notebooks/seq2seq-vs-GPR-Lorenz-prediction-tensorflow-xyz2xyz.ipynb) | [Lorenz.mat](https://github.com/suzyi/recurrent-neural-network/tree/master/data/Lorenz.mat) |
| Jun 12, 2019 | [wildml-define-rnn-using-numpy](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/) | |
| Jun 11, 2019 | [copytask-seq2seq-on-tensorflow-for-beginner.ipynb](https://github.com/suzyi/recurrent-neural-network/blob/master/notebooks/copytask-seq2seq-on-tensorflow-for-beginner.ipynb) | |
| Jun 11, 2019 | [Intro-to-LSTM-and-Seq2seq.ipynb](https://github.com/suzyi/recurrent-neural-network/blob/master/notebooks/Intro-to-LSTM-and-Seq2seq.ipynb)| |
| May 15, 2019 | [Predict-KS-via-Reservoir-Computing-all2all.ipynb](https://github.com/suzyi/recurrent-neural-network/blob/master/notebooks/Predict-KS-via-Reservoir-Computing-all2all.ipynb)| [KS.mat](https://github.com/suzyi/recurrent-neural-network/tree/master/data/KS.mat) |
| May 15, 2019 | [lstm-Lorenz-prediction-xyz2xyz-batch-training-on-tensorflow.ipynb](https://github.com/suzyi/recurrent-neural-network/blob/master/notebooks/lstm-Lorenz-prediction-xyz2xyz-batch-training-on-tensorflow.ipynb) | [Lorenz.mat](https://github.com/suzyi/recurrent-neural-network/tree/master/data/Lorenz.mat)|
| May 13, 2019 | [rnn-mnist.ipynb](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/recurrent_network.ipynb) | |
| May 13, 2019 | [define_lstm_unit_from_scratch_tensorflow.ipynb](https://github.com/suzyi/recurrent-neural-network/blob/master/notebooks/define_lstm_unit_from_scratch_tensorflow.ipynb) | |
| May 13, 2019 | [seq2seq-sine-cosine-prediction-tensorflow.ipynb](https://github.com/suzyi/recurrent-neural-network/blob/master/notebooks/seq2seq-sine-cosine-prediction-tensorflow.ipynb) | |
| May 13, 2019 | [Predict-Lorenz-using-Reservoir-Computing-simple-demo.ipynb](https://github.com/suzyi/recurrent-neural-network/blob/master/notebooks/Predict-Lorenz-using-Reservoir-Computing-simple-demo.ipynb) | [Lorenz.mat](https://github.com/suzyi/recurrent-neural-network/tree/master/data/Lorenz.mat)|


## 4 - References
**Websites**
+ [wikipedia-Recurrent_neural_network](https://en.wikipedia.org/wiki/Recurrent_neural_network)
+ Tensorflow based coding guideline for RNN - [easy tensorflow: RNNs](http://www.easy-tensorflow.com/tf-tutorials/recurrent-neural-networks/)

**Slides**
+ Lecture given by Fei-Fei Li & Justin Johnson & Serena Yeung - [stanford cs231n slides](http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture10.pdf)

**Literatures**
+ *Reservoir Computing for chaotic KS prediction.* Pathak, Jaideep, et al. "Model-free prediction of large spatiotemporally chaotic systems from data: A reservoir computing approach." Physical review letters 120.2 (2018): 024102.
+ *Echo States Networks for time series prediction.* Jaeger, Herbert, and Harald Haas. "Harnessing nonlinearity: Predicting chaotic systems and saving energy in wireless communication." science 304.5667 (2004): 78-80.
+ *Chapter 10 Sequence Modeling: Recurrent and Recursive Nets.* LeCun, Yann, Yoshua Bengio, and Geoffrey Hinton. "Deep learning." nature 521.7553 (2015): 436.
+ *NIPS Time Series Workshop 2017.* Yu, Rose, et al. "Long-term forecasting using tensor-train rnns." arXiv preprint arXiv:1711.00073 (2017).
