import numpy as np
import pytest
from nn.nn import NeuralNetwork  # This should work now
from nn import preprocess

# ...rest of your code
def test_single_forward():
    nn = NeuralNetwork(
        nn_arch=[{"input_dim": 2, "output_dim": 2, "activation": "relu"}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=1,
        loss_function="mean_squared_error",
    )

    X = np.array([[1, 1]])
    b_curr = nn._param_dict["b1"] # bias for layer 1
    W_curr = nn._param_dict["W1"] # weights for layer 1

    activation = nn.arch[0]["activation"]
    A, Z = nn._single_forward(W_curr, b_curr, X, activation)

    assert A.shape == (1, 2) #dimension check
    assert Z.shape == (2, 1) #dimension check
    assert np.all(nn._relu(A).T == Z)

def test_forward():
    nn = NeuralNetwork(
        nn_arch=[{"input_dim": 2, "output_dim": 2, "activation": "relu"}, {"input_dim": 2, "output_dim": 1, "activation": "relu"}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=1,
        loss_function="mean_squared_error",
    )

    X = np.array([[1, 1]])

    A, cache = nn.forward(X)

    assert A.shape == (1, 1)

    assert cache["A1"].shape == (1, 2)
    assert cache["A2"].shape == (1, 1)
    assert cache["Z1"].shape == (2, 1)
    assert cache["Z2"].shape == (1, 1)

def test_single_backprop():
    nn = NeuralNetwork(
        nn_arch=[{"input_dim": 2, "output_dim": 2, "activation": "relu"}, {"input_dim": 2, "output_dim": 1, "activation": "relu"}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=1,
        loss_function="mean_squared_error",
    )
    
    X = np.array([[2, 2]])
    Y = np.array([[1]])

    b = nn._param_dict["b2"]
    W = nn._param_dict["W2"]

    activation = nn.arch[1]["activation"]

    A, cache = nn.forward(X)

    dA = nn._mean_squared_error_backprop(Y, A)

    Z = cache["Z2"]
    A_prev = cache["A1"]

    dA_prev, dW, db = nn._single_backprop(W, b, Z, A_prev, dA, activation)

    assert dA_prev.shape == (1, 2)
    assert dW.shape == (1, 2)
    assert db.shape == (1, 1)

def test_predict():
    nn = NeuralNetwork(
        nn_arch=[{"input_dim": 3, "output_dim": 3, "activation": "relu"}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=1,
        loss_function="mean_squared_error",
    )

    nn._param_dict["W1"] = np.array([[0, 0], [1, 0], [0, 1]])
    nn._param_dict["b1"] = np.array([[0], [1], [0]])

    X = np.array([[1, 1]])
    Y = nn.predict(X)

    Y = nn._relu(Y)

    assert Y.shape == (1, 3)
    assert np.all(Y == np.array([[0, 2, 1]]))

def test_binary_cross_entropy():
    nn = NeuralNetwork(
        nn_arch=[{"input_dim": 3, "output_dim": 3, "activation": "relu"}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=1,
        loss_function="mean_squared_error",
    )

    y_hat = np.array([0.9, 0.1, 0.1])
    y = np.array([1, 0, 0])

    loss = nn._binary_cross_entropy(y, y_hat)
    exp_loss = np.mean([-np.log(0.9), -np.log(0.9), -np.log(0.9)])

    assert np.allclose(loss, exp_loss, atol=0.0001)

def test_binary_cross_entropy_backprop():
    nn = NeuralNetwork(
        nn_arch=[{"input_dim": 3, "output_dim": 3, "activation": "relu"}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=1,
        loss_function="binary_cross_entropy",
    )

    y_hat = np.array([0.9, 0.1, 0.1])
    y = np.array([1, 0, 0])

    dA = nn._binary_cross_entropy_backprop(y, y_hat)

    exp_dA = np.array([-1/0.9, 1/0.9, 1/0.9])

    assert np.allclose(dA, exp_dA, atol=0.0001)

def test_mean_squared_error():
    nn = NeuralNetwork(
        nn_arch=[{"input_dim": 3, "output_dim": 3, "activation": "relu"}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=1,
        loss_function="mean_squared_error",
    )

    y_hat = np.array([0.9, 0.1, 0.1])
    y = np.array([1, 0, 0])

    loss = nn._mean_squared_error(y, y_hat)
    exp_loss = np.mean([0.1**2, 0.1**2, 0.1**2])

    assert np.allclose(loss, exp_loss, atol=0.0001)

def test_mean_squared_error_backprop():
    nn = NeuralNetwork(
        nn_arch=[{"input_dim": 3, "output_dim": 3, "activation": "relu"}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=1,
        loss_function="mean_squared_error",
    )

    y_hat = np.array([0.9, 0.1, 0.1])
    y = np.array([1, 0, 0])

    dA = nn._mean_squared_error_backprop(y, y_hat)

    exp_dA = np.array([2*-0.1, 2*0.1, 2*0.1])

    assert np.allclose(dA, exp_dA, atol=0.0001)

def test_sample_seqs():
    seqs = ["AA", "CC", "TT", "GG", "AAA", "CCC", "TTT", "GGG"]
    labels = [0, 0, 1, 0, 1, 1, 0, 1]
    sampled_seqs, sampled_labels = preprocess.sample_seqs(seqs, labels)

    assert len(sampled_seqs) == len(sampled_labels), f"ERROR {len(sampled_seqs)} != {len(sampled_labels)}"

def test_one_hot_encode_seqs():
    sequences = ["AA", "CC", "TT", "GG"]

    encodings = preprocess.one_hot_encode_seqs(sequences)

    assert encodings.shape == (4, 8)
