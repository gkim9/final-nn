# NOTE: I ran into a bunch of issues with dimensions not working, so I had to resort to AI tools to debug. 

# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """

        Z_curr = np.dot(W_curr, A_prev.T) + b_curr

        if activation == "relu":
            A_curr = self._relu(Z_curr)
        elif activation == "sigmoid":
            A_curr = self._sigmoid(Z_curr)
        else:
            raise Exception(f"'{activation}' not supported")
        
        return A_curr.T, Z_curr


    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        cache = {"A0": X}
        A_curr = X

        for idx, layer in enumerate(self.arch):
            idx += 1
            A_prev = A_curr

            W_curr = self._param_dict["W" + str(idx)]
            b_curr = self._param_dict["b" + str(idx)]

            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, layer["activation"])

            cache["A" + str(idx)] = A_curr
            cache["Z" + str(idx)] = Z_curr

        return A_curr, cache

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        if activation_curr == 'relu':
            dZ_curr = self._relu_backprop(dA_curr.T, Z_curr)

        elif activation_curr == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr.T, Z_curr)

        else:
            raise Exception(f"'{activation_curr}' not supported")
        
        m = A_prev.shape[0] # get mean across batch, need size of batch

        dA_prev = np.dot(W_curr.T, dZ_curr).T # from dZ/dA = W_curr, transposed to match shapes
        dW_curr = np.dot(dZ_curr, A_prev) / m # from dZ/dW = A_prev
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m # from dZ/db = 1 (and summing each row)

        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        grad_dict = {}

        y = y.reshape(y_hat.shape)

        if self._loss_func == "binary_cross_entropy":
            dA_prev = self._binary_cross_entropy_backprop(y, y_hat)
        
        elif self._loss_func == "mean_squared_error":
            dA_prev = self._mean_squared_error_backprop(y, y_hat)

        else:
            raise Exception(f"'{self._loss_func}' is not supported")
        
        for layer_idx in range(len(self.arch), 0, -1):  # This is equivalent to reversed(range(1, len(self.arch) + 1))
            # For the current layer, store the gradient
            dA_curr = dA_prev
            A_prev = cache["A" + str(layer_idx - 1)]
            Z_curr = cache["Z" + str(layer_idx)]
            W_curr = self._param_dict["W" + str(layer_idx)]
            b_curr = self._param_dict["b" + str(layer_idx)]
            activation_curr = self.arch[layer_idx - 1]["activation"]

            # Backpropagate through the layer to compute gradients
            dA_prev, dW_curr, db_curr = self._single_backprop(
                W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr
            )

            # Save the gradients for this layer
            grad_dict["dW" + str(layer_idx)] = dW_curr
            grad_dict["db" + str(layer_idx)] = db_curr

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for idx, layer in enumerate(self.arch):
            idx += 1
            self._param_dict["W" + str(idx)] -= self._lr * grad_dict['dW' + str(idx)]
            self._param_dict["b" + str(idx)] -= self._lr * grad_dict['db' + str(idx)]

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        if self._loss_func == "binary_cross_entropy":
            loss_function = self._binary_cross_entropy
        elif self._loss_func == "mean_squared_error":
            loss_function = self._mean_squared_error
        else:
            raise Exception(f"'{self._loss_func}' not supported")
        
        for epoch in range(self._epochs):
            for i in range(0, X_train.shape[0], self._batch_size):
                X_batch = X_train[i:i + self._batch_size]
                y_batch = y_train[i:i + self._batch_size]

                y_hat, cache = self.forward(X_batch)

                grad_dict = self.backprop(y_batch, y_hat, cache)
                self._update_params(grad_dict)
            
            y_hat_train = self.predict(X_train)
            loss_train = loss_function(y_train, y_hat_train)

            per_epoch_loss_train.append(loss_train)

            #validate
            y_hat_val = self.predict(X_val)
            loss_val = loss_function(y_val, y_hat_val)

            per_epoch_loss_val.append(loss_val)
        
        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, cache = self.forward(X)

        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1 / (1 + np.exp(-Z))

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        Z_input = self._sigmoid(Z)
        return dA * Z_input * (1 - Z_input)

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return Z * (Z > 0) # return Z if Z>0 and 0 if Z<=0

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        return (Z > 0) * dA # return dA*Z if Z>0 and 0 if Z<=0

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """

        return np.mean(-1 * y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)).item()

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """

        return - y / y_hat + (1 - y) / (1 - y_hat)

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """

        return np.mean((y - y_hat) ** 2)

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """

        return 2 * (y_hat - y)