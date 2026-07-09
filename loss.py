from layer import Layer
import numpy as np

class Loss(Layer):
    def forward(self, predicted: np.array, real: np.array) -> float:
        """
        Calculates the MSE for the given batch
        """
        return np.mean((predicted - real)**2)
    
    def backward(self, predicted: np.array, real: np.array):
        """
        Calculates the derivative of MSE for the given batch
        """

        return np.mean((2/predicted.size) * (predicted - real), axis=0)