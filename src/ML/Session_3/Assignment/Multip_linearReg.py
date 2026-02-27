import numpy as np 
import matplotlib.pyplot as plt


class MultipLinearRegression_Hazem:
    """ 

                                """

    def __init__(self, alpha=0.01, num_iterations=100):
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.w = None   # Now will be vector
        self.b = 0
        self.SSE_values = []
        self.MSE_values = []

    def initialize_parameters(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def compute_gradients(self, X, y, y_hat):
        n = len(y)
        dw = (1/n) * np.dot(X.T, (y_hat - y))
        db = (1/n) * np.sum(y_hat - y)
        return dw, db

    def compute_sse(self, y, y_hat):
        return np.sum((y_hat - y) ** 2)
    
    def compute_mse(self, y, y_hat):
        return np.mean((y_hat - y) ** 2)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.initialize_parameters(n_features)

        for i in range(self.num_iterations):

            y_hat = self.predict(X)

            dw, db = self.compute_gradients(X, y, y_hat)

            self.w -= self.alpha * dw
            self.b -= self.alpha * db

            # re-compute predictions
            y_hat = self.predict(X)

            sse = self.compute_sse(y, y_hat)
            mse = self.compute_mse(y, y_hat)

            self.SSE_values.append(sse)
            self.MSE_values.append(mse)

            if (i + 1) % 20 == 0:
                print(f"Iteration {i+1}, SSE: {sse}, MSE: {mse}")

        return self

    def plot_loss(self):
        plt.figure(figsize=(6, 4))
        plt.plot(range(self.num_iterations), self.MSE_values)
        plt.xlabel("Iteration")
        plt.ylabel("MSE")
        plt.title("MSE over Iterations")
        plt.show()

    def get_params(self):
        return {
            "weights": self.w,
            "bias": self.b
        }
