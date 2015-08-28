import numpy as np
import theano
import theano.tensor as T

def one_hot(string):
    """
    Take a string and return a one-hot encoding with ASCII
    """
    res = np.zeros((len(string), 256))
    for i in xrange(len(string)):
        res[i,ord(string[i])] = 1.
    return res

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def random_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def zeros(shape):
    return theano.shared(floatX(np.zeros(shape)))

def softmax(X, temperature=1.0):
    e_x = T.exp((X - X.max(axis=1).dimshuffle(0, 'x'))/temperature)
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

# def softmax(X, temperature):
#     e_x = T.exp((X-X.max())/temperature)
#     return e_x / e_x.sum()

def sigmoid(X):
    return 1 / (1 + T.exp(-X))


def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
        return X


def rectify(X):
    return T.maximum(X, 0.)


def SGD (cost, params, eta):
    updates = []
    grads = T.grad(cost=cost, wrt=params)

    for p,g in zip(params, grads):
        updates.append([p, p - eta * g])

    return updates


def momentum(cost, params, caches, eta, rho=.1):
    updates = []
    grads = T.grad(cost=cost, wrt=params)

    for p, c, g in zip(params, caches, grads):
        delta = rho * g + (1-rho) * c
        updates.append([c, delta])
        updates.append([p, p - eta * delta])

    return updates


def sample_char(probs):
    return one_hot_to_string(np.random.multinomial(1, probs))


def one_hot_to_string(one_hot):
    return chr(one_hot.nonzero()[0][0])

def get_params(layers):
    params = []
    for layer in layers:
        params += layer.get_params()
    return params

def zeros(length):
    return theano.shared(floatX(np.zeros(length)))

def make_caches(params):
    caches = []
    for p in params:
        caches.append(theano.shared(floatX(np.zeros(p.get_value().shape))))

    return caches

def one_step_updates(layers):
    updates = []

    for layer in layers:
        updates += layer.updates()

    return updates
