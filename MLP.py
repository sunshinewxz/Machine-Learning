"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

>>> activation = Identity()
>>> activation(3)
3
>>> activation.forward(3)
3
"""

import numpy as np
import os


def load_mnist_data_file(path):
    test_data = np.load(path+'/test_data.npy')
    test_labels = np.load(path+'/test_labels.npy')
    train_data = np.load(path+'/train_data.npy')
    train_labels = np.load(path+'/train_labels.npy')
    val_data = np.load(path+'/val_data.npy')
    val_labels = np.load(path+'/val_labels.npy')
    return test_data,test_labels,train_data,train_labels,val_data,val_labels


class Activation(object):
    """ Interface for activation functions (non-linearities).

        In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    def __init__(self):
        self.state = None
        self.input = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        raise NotImplemented


class Identity(Activation):
    """ Identity function (already implemented).
     """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.input = x
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):
    """ Implement the sigmoid non-linearity """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.input = x
        x_sig = 1/(1+np.exp(-x))
        self.state = x_sig
        return x_sig

    def derivative(self):
        x_deri = (self.state)*(1-self.state)
        return x_deri


class Tanh(Activation):
    """ Implement the tanh non-linearity """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.input = x
        x_1 = np.exp(x) - np.exp(-x)
        x_2 = np.exp(x) + np.exp(-x)
        x_tanh = x_1/x_2
        self.state = x_tanh
        return x_tanh

    def derivative(self):
        x_deri = 1-np.square(self.state)
        return x_deri


class ReLU(Activation):
    """ Implement the ReLU non-linearity """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        x_relu = np.maximum(0,x)
        self.state = x_relu
        self.input = x
        return x_relu

    def derivative(self):
        x,y = (self.input).shape
        x_deri = np.full((x,y),0)
        x_deri[(self.input>0)] = 1
        return x_deri


# CRITERION


class Criterion(object):
    """ Interface for loss functions.
    """

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):
    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = []
        self.loss = []

    def forward(self, x, y):
        for i in range(len(x)):
            a = np.max(x[i])
            ex = np.exp(x[i]-a)
            sm = ex/np.sum(ex)
            sm_log = np.log(sm)
            li = y[i].dot(sm_log.T)
            li_fu = li * (-1)
            (self.sm).append(sm)
            (self.loss).append(np.sum(li_fu))
        self.labels = y
        return self.loss

    def derivative(self):
        deri = []
        for i in range(len(self.sm)):
            deri.append(self.sm[i] - self.labels[i])
        return deri


class BatchNorm(object):
    def __init__(self, fan_in, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        m = x.shape[0]
        if eval == False:
            self.mean = np.mean(x,axis=0)
            self.var = np.var(x,axis=0)
            self.x = x
            self.norm = (x - self.mean)/np.sqrt(self.var+self.eps)
            self.out = (self.gamma) * (self.norm) + self.beta
            self.running_mean = self.alpha * self.running_mean + (1-self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1-self.alpha) * self.var
        else:
            temp = (x-self.running_mean)/np.sqrt(self.running_var + self.eps)
            self.out = self.gamma * temp + self.beta
        return self.out


    def backward(self, delta):
        m = (self.x).shape[0]
        dnorm = delta * self.gamma
        dnorm_var = (-1 / 2) * (self.x - self.mean) * (self.var + self.eps) ** (-3 / 2)
        dvar = np.sum(dnorm * dnorm_var, axis=0)
        dx1 = dnorm*(self.var+self.eps)**(-1/2)
        dx2 = dvar*(2/m*(self.x-self.mean))
        dmean = (-1)*np.sum(dnorm*(self.var+self.eps)**(-1/2),axis=0)-2/m*dvar*np.sum(self.x-self.mean,axis=0)
        dx3 = dmean/m
        dx = dx1 + dx2 + dx3
        self.dbeta = m*np.sum(delta,axis=0)
        self.dgamma = m*np.sum(delta * self.norm, axis=0)
        return dx


def random_normal_weight_init(d0, d1):
    w = np.random.randn(d0, d1)
    return w

def zeros_bias_init(d):
    b = np.full((1,d),0)
    return b


class MLP(object):
    """ A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens,
                 activations, weight_init_fn, bias_init_fn,
                 criterion, lr, momentum=0.0, num_bn_layers=0):
        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes
        self.W = None
        self.dW = None
        self.b = None
        self.db = None
        # if batch norm, add batch norm parameters
        self.bn_layers = []
        if self.bn:
            for i in range(self.num_bn_layers):
                self.bn_layers.append(BatchNorm(hiddens[i]))

        # Feel free to add any other attributes useful to your implementation (input, output, ...)
        self.input = None
        self.hiddens = hiddens
        self.vw = []
        self.vb = []

        self.W = []
        self.b = []
        self.dW = []
        self.db = []

        if(self.nlayers > 1):
            (self.W).append(weight_init_fn(input_size,self.hiddens[0]))
            (self.b).append(bias_init_fn(self.hiddens[0]))
            (self.vw).append(np.full((input_size,self.hiddens[0]),0))
            (self.vb).append(np.full((1,self.hiddens[0]),0))
            for i in range(1,self.nlayers-1):
                (self.W).append(weight_init_fn(self.hiddens[i-1],self.hiddens[i]))
                (self.b).append(bias_init_fn(self.hiddens[i]))
                (self.vw).append(np.full((self.hiddens[i-1], self.hiddens[i]),0))
                (self.vb).append(np.full((1, self.hiddens[i]),0))
            (self.W).append(weight_init_fn(self.hiddens[self.nlayers-2],output_size))
            (self.b).append(bias_init_fn(output_size))
            (self.vw).append(np.full((self.hiddens[self.nlayers-2], output_size), 0))
            (self.vb).append(np.full((1, output_size), 0))
        else:
            (self.W).append(weight_init_fn(input_size, output_size))
            (self.b).append(bias_init_fn(output_size))
            (self.vw).append(np.full((input_size, output_size), 0))
            (self.vb).append(np.full((1, output_size), 0))

    def forward(self, x):
        self.input = x
        for i in range(self.nlayers):
            w = self.W[i]
            b = self.b[i]
            y = np.dot(x, w) + b
            if (i < self.num_bn_layers):
                y = self.bn_layers[i].forward(y,not self.train_mode)
            x = (self.activations[i]).forward(y)
        return x

    def zero_grads(self):
        self.db = self.db *0
        self.dW = self.dW *0

    def step(self):
        for i in range(len(self.W)):
            self.vw[i] = self.momentum * self.vw[i] - self.lr * self.dW[i]
            self.vb[i] = self.momentum * self.vb[i] - self.lr * self.db[i]
            self.W[i] = self.W[i] + self.vw[i]
            self.b[i] = self.b[i] + self.vb[i]
        for i in range(self.num_bn_layers):
            (self.bn_layers[i]).beta = (self.bn_layers[i]).beta - self.lr * (self.bn_layers[i]).dbeta
            (self.bn_layers[i]).gamma = (self.bn_layers[i]).gamma - self.lr * (self.bn_layers[i]).dgamma

    def backward(self, labels):
        sc = self.criterion
        act = self.activations[self.nlayers - 1]
        y = act.state
        err = sc.forward(y, labels)
        d = sc.derivative()/len(labels)
        for i in range(self.nlayers-1):
            act = self.activations[self.nlayers-i-1]
            d = d * act.derivative()
            if (self.nlayers - i - 1) < self.num_bn_layers:
                d = self.bn_layers[i].backward(d)
            yl = (self.activations[self.nlayers-i-2]).state
            (self.dW).append(np.dot(yl.T,d))
            (self.db).append(np.sum(d,axis=0))
            w = self.W[self.nlayers-i-1]
            d = np.dot(d,w.T)
        act = self.activations[0]
        d = d * act.derivative()
        if self.num_bn_layers > 0:
            d = self.bn_layers[0].backward(d)
        y_inp = self.input
        (self.dW).append(np.dot(y_inp.T, d))
        (self.db).append(np.sum(d, axis=0))
        (self.dW).reverse()
        (self.db).reverse()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):
    train_data, train_labels = dset[0]
    val_data, val_labels = dset[1]
    test_data, test_labels = dset[2]
    training_losses = np.full((1,nepochs),0)
    training_errors = np.full((1,nepochs),0)
    validation_losses = np.full((1,nepochs),0)
    validation_errors = np.full((1,nepochs),0)
    batch_num_train = train_data.shape[0]/batch_size
    batch_num_val = val_data.shape[0]/batch_size
    for i in range(nepochs):
        training = np.hstack((train_data,train_labels))
        np.random.shuffle(training)
        train_data = training[:,0:784]
        train_labels = training[:,784:794]
        loss = 0
        error = 0
        for j in range(0,train_data.shape[0],batch_size):
            mlp.zero_grads()
            batch_train_data = train_data[j:j+batch_size,:]
            batch_train_labels = train_labels[j:j+batch_size,:]
            pre = mlp.forward(batch_train_data)
            sc = SoftmaxCrossEntropy()
            loss += np.sum(sc.forward(pre, batch_train_labels))
            pre_pos = np.argmax(pre,axis=1)
            pre_label = np.full(batch_train_labels.shape,0)
            for k in range(len(pre_pos)):
                pre_label[k,pre_pos[k]] = 1
            error += np.sum(np.abs(pre_label-batch_train_labels))
            mlp.backward(batch_train_labels)
            mlp.step()
        loss_val = 0
        error_val = 0
        for j in range(0,val_data.shape[0],batch_size):
            batch_val_data = val_data[j:j + batch_size, :]
            batch_val_labels = val_labels[j:j + batch_size, :]
            pre_val = mlp.forward(batch_val_data)
            sc_val = SoftmaxCrossEntropy()
            loss_val += np.sum(sc_val.forward(pre_val,batch_val_labels))
            pre_pos = np.argmax(pre_val,axis=1)
            pre_label = np.full(batch_val_labels.shape, 0)
            for k in range(len(pre_pos)):
                pre_label[k, pre_pos[k]] = 1
            error_val += np.sum(np.abs(pre_label - batch_val_labels))

        training_losses[0,i] = loss/(batch_num_train * batch_size)
        training_errors[0,i] = error/(2 * batch_num_train * batch_size)
        validation_losses[0,i] = loss_val/(batch_num_val * batch_size)
        validation_errors[0,i] = error_val/(2 * batch_num_val * batch_size)

    confusion_matrix = np.full((10,10),0)
    for i in range(0, test_data.shape[0], batch_size):
        batch_test_data = test_data[i:i + batch_size, :]
        batch_test_labels = test_labels[i:i + batch_size, :]
        pre_test = mlp.forward(batch_test_data)
        pre_pos = np.argmax(pre_test, axis=1)
        m,true_pos = np.where(batch_test_labels==1)
        for j in range(batch_test_labels.shape[0]):
            confusion_matrix[pre_pos[j]][true_pos[j]] += 1

    return (training_losses,training_errors,validation_losses,validation_errors,confusion_matrix)





