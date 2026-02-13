import numpy as np
import matplotlib.pyplot as plt

class MultipLinearRegression_Hazem:

    def __init__(self, alpha=0.01, num_iterations=100):
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.w = None          # weight vector
        self.b = 0
        self.SSE_values = []
        self.MSE_values = []

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def compute_gradients(self, X, y, y_hat):
        n = len(y)
        dw = np.mean((y_hat - y).reshape(-1, 1) * X, axis=0)
        db = np.mean(y_hat - y)
        return dw, db

    def compute_sse(self, y, y_hat):
        return np.sum((y_hat - y) ** 2)

    def compute_mse(self, y, y_hat):
        return np.mean((y_hat - y) ** 2)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        for i in range(self.num_iterations):
            y_hat = self.predict(X)

            dw, db = self.compute_gradients(X, y, y_hat)

            self.w -= self.alpha * dw
            self.b -= self.alpha * db

            y_hat = self.predict(X)

            self.SSE_values.append(self.compute_sse(y, y_hat))
            self.MSE_values.append(self.compute_mse(y, y_hat))

            if (i + 1) % 20 == 0:
                print(
                    f"Iteration {i+1}, "
                    f"SSE: {self.SSE_values[-1]}, "
                    f"MSE: {self.MSE_values[-1]}"
                )

        return self

    def get_params(self):
        return {
            "weights": self.w,
            "bias": self.b
        }
