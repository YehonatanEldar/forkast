from layer import Layer
import numpy as np

class Loss(Layer):
    def forward(predicted: np.array, real: np.array) -> float:
        """
        Calculates the MSE for the given batch
        """

        return np.mean((predicted - real)**2)    