import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import gzip
import pickle
from PIL import Image
import random
import numpy as np
from sklearn.utils import shuffle

def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")

def download_mnist():
    os.mkdir(dataset_dir)
    for v in key_file.values():
       _download(v)

def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data

def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """
    Parameters
    ----------
    normalize : Normalize the pixel values
    flatten : Flatten the images as one array
    one_hot_label : Encode the labels as a one-hot array

    Returns
    -------
    (Trainig Image, Training Label), (Test Image, Test Label)
    """
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


# Load the MNIST dataset
url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

# dataset_dir = os.path.dirname(os.path.abspath(__file__))

dataset_dir = "data"
save_file = dataset_dir + "/mnist.pkl"
np.random.seed(0)
train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))

    return sig

def sigmoid_grad(x):
    sig_grad = sigmoid(x) * (1 - sigmoid(x))

    return sig_grad

def softmax(x):
    """
  Softmax loss function


  Inputs:
  - X: A numpy array of shape (N, C) containing a minibatch of data.
  Returns:
  - probabilities: A numpy array of shape (N, C) containing the softmax probabilities.
     """
    probabilities = (np.exp(x.T) / np.sum(np.exp(x.T), axis=0)).T

    return probabilities

def cross_entropy_error(y, t):
    """
    Inputs:

    - t:  A numpy array of shape (N,C) containing  a minibatch of training labels, it is a one-hot array,
      with t[GT]=1 and t=0 elsewhere, where GT is the ground truth label ;
    - y: A numpy array of shape (N, C) containing the softmax probabilities (the NN's output).

    Returns a tuple of:
    - loss as single float
    """
    # Compute loss
    m = y.shape[0]
    log_likelihood = -np.log(y[range(m), t.argmax(axis=1)])
    error = np.sum(log_likelihood) / m
    return error

def TwoLayerNet(input_size, hidden_size, output_size, weight_init_std=0.01):
    params = {}
    params['b1'] = np.zeros(hidden_size)
    params['W1'] = np.random.normal(0, weight_init_std, (input_size, hidden_size))
    params['b2'] = np.zeros(output_size)
    params['W2'] = np.random.normal(0, weight_init_std, (hidden_size, output_size))

    return params

def FC_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.
    The input x has shape (N, D) and contains a minibatch of N
    Inputs:
    - x: A numpy array containing input data, of shape (N, D)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output result of the forward pass, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    out = x.reshape(x.shape[0], w.shape[0]).dot(w) + b
    cache = (x, w, b)
    return out, cache

def FC_backward(dout, cache):
    """
    Computes the backward pass for a fully-connected layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
    - w: Weights, of shape (D, M)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, D)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], w.shape[0]).T.dot(dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db

def Model(params, x, t):
    """
    Computes the backward pass for a fully-connected layer.
    Inputs:
    - params:  dictionary with first layer weights and biases using the keys 'W1' and 'b1' and second layer weights
    and biases using the keys 'W2' and 'b2'. each with dimensions corresponding its input and output dimensions.
    - x: Input data, of shape (N,D)
    - t:  A numpy array of shape (N,C) containing training labels, it is a one-hot array,
      with t[GT]=1 and t=0 elsewhere, where GT is the ground truth label ;
    Returns:
    - y: the output probabilities for the minibatch (at the end of the forward pass) of shape (N,C)
    - grads: dictionary containing gradients of the loss with respect to W1, W2, b1, b2.

    """
    W1, W2 = params['W1'], params['W2']
    b1, b2 = params['b1'], params['b2']
    grads = {'W1': None, 'W2': None, 'b1': None, 'b2': None}

    batch_num = x.shape[0]

    m = batch_num
    # forward (fullyconnected -> sigmoid -> fullyconnected -> softmax).
    z1, x_cache = FC_forward(x, W1, b1)
    a1 = sigmoid(z1)
    z2, a1_cache = FC_forward(a1, W2, b2)
    y = softmax(z2)
    # backward - calculate gradients.
    # softmax grad
    dz2 = y - t  # CE error in output
    da1, dw, db = FC_backward(dz2, a1_cache)
    grads['W2'], grads['b2'] = dw/m, db/m
    dz1 = da1*sigmoid_grad(z1)
    _, dw, db  = FC_backward(dz1, x_cache)
    grads['W1'], grads['b1'] = dw/m, db/m

    return grads, y

def accuracy(y, t):
    """
    Computes the accuracy of the NN's predictions.
    Inputs:
    - t:  A numpy array of shape (N,C) containing training labels, it is a one-hot array,
      with t[GT]=1 and t=0 elsewhere, where GT is the ground truth label ;
    - y: the output probabilities for the minibatch (at the end of the forward pass) of shape (N,C)
    Returns:
    - accuracy: a single float of the average accuracy.
    """
    correct = 0
    for i in range(y.shape[0]):
        if (y[i] >= 0.5).any():
            pred = y[i].argmax(axis=0)
            if pred == t[i].argmax(axis=0):
                correct += 1
    accuracy = 100*correct / y.shape[0]

    return accuracy

epochs = 10
mini_batch_size = 100
learning_rate = 4
num_hidden_cells = 64
np.random.seed(0)

def Train(epochs_num, batch_size, lr, H):
    #  Dividing a dataset into training data and test data

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    C = 10
    D = x_train.shape[1]
    network_params = TwoLayerNet(input_size=D, hidden_size=H,
                                 output_size=C)  # hidden_size is the only hyperparameter here

    train_size = x_train.shape[0]
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    iter_per_epoch = round(train_size / batch_size)

    print('training of ' + str(epochs_num) + ' epochs, each epoch will have ' + str(iter_per_epoch) + ' iterations')
    for i in range(epochs_num):

        train_loss_iter = []
        train_acc_iter = []

        for k in range(iter_per_epoch):

            try:  # make it run faster
                shuffle_index
            except NameError:
                shuffle_index = np.random.permutation(x_train.shape[0])
                X_train_shuffled = x_train[shuffle_index, :]
                Y_train_shuffled = t_train[shuffle_index, :]
            begin = k * batch_size
            end = min(begin + batch_size, x_train.shape[0] - 1)
            x_batch = X_train_shuffled[begin:end,:]
            t_batch = Y_train_shuffled[begin:end,:]

            grads_batch, y_batch = Model(network_params, x_batch, t_batch)

            network_params['W1'] -= lr * grads_batch['W1']
            network_params['b1'] -= lr * grads_batch['b1']
            network_params['W2'] -= lr * grads_batch['W2']
            network_params['b2'] -= lr * grads_batch['b2']

            # Calculate the loss and accuracy for visalizaton

            error = cross_entropy_error(y_batch, t_batch)
            train_loss_iter.append(error)
            acc_iter = accuracy(y_batch, t_batch)
            train_acc_iter.append(acc_iter)
            if k == iter_per_epoch - 1:
                train_acc = np.mean(train_acc_iter)
                train_acc_list.append(train_acc)
                train_loss_list.append(np.mean(train_loss_iter))

                _, y_test = Model(network_params, x_test, t_test)
                test_acc = accuracy(y_test, t_test)
                test_acc_list.append(test_acc)
                print("train acc: " + str(train_acc)[:5] + "% |  test acc: " + str(
                    test_acc) + "% |  loss for epoch " + str(i) + ": " + str(np.mean(train_loss_iter)))
    return train_acc_list, test_acc_list, train_loss_list, network_params

train_acc, test_acc, train_loss, net_params = Train(epochs, mini_batch_size, learning_rate, num_hidden_cells)

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc))
plt.plot(x, train_acc, label='train acc')
plt.plot(x, test_acc, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(loc='lower right')
plt.show()

markers = {'train': 'o'}
x = np.arange(len(train_loss))
plt.plot(x, train_loss, label='train loss')
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.legend(loc='lower right')
plt.show()

# Visualize some weights. features of digits should be somehow present.
def show_net_weights(params):
    W1 = params['W1']
    print(W1.shape)
    for i in range(5):
        W = W1[:,i*5].reshape(28, 28)
        plt.imshow(W,cmap='gray')
        plt.axis('off')
        plt.show()

show_net_weights(net_params)
