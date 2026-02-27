import numpy as np
import matplotlib.pyplot as plt

class LinearRegression_Hazem:

    def __init__(self, alpha=0.01, num_iterations=100):
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.w = 0
        self.b = 0
        self.lambda_ = 0.0
        self.reg_type = "none"
        self.SSE_values = []
        self.MSE_values = []

    def predict(self, x):
        return self.w * x + self.b

    def compute_gradients(self, x, y, y_hat):
        dw = np.mean((y_hat - y) * x)
        db = np.mean(y_hat - y)

        if self.reg_type == "ridge":        # L2
            dw += self.lambda_ * self.w

        elif self.reg_type == "lasso":      # L1
            dw += self.lambda_ * np.sign(self.w)

        return dw, db

    def compute_sse(self, y, y_hat):
        return np.sum((y_hat - y) ** 2)

    def compute_mse(self, y, y_hat):
        return np.mean((y_hat - y) ** 2)

    def fit(self, x, y):
        self.SSE_values = []
        self.MSE_values = []

        for i in range(self.num_iterations):
            y_hat = self.predict(x)

            dw, db = self.compute_gradients(x, y, y_hat)

            self.w -= self.alpha * dw
            self.b -= self.alpha * db

            y_hat = self.predict(x)

            self.SSE_values.append(self.compute_sse(y, y_hat))
            self.MSE_values.append(self.compute_mse(y, y_hat))

        return self

    def fit_ridge(self, x, y, lambda_):
        self.lambda_ = lambda_
        self.reg_type = "ridge"
        return self.fit(x, y)

    def fit_lasso(self, x, y, lambda_):
        self.lambda_ = lambda_
        self.reg_type = "lasso"
        return self.fit(x, y)

    def plot(self, x, y):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.SSE_values)
        plt.xlabel("Iteration")
        plt.ylabel("SSE")
        plt.title("SSE over Iterations")

        plt.subplot(1, 2, 2)
        plt.scatter(x, y, label="Data")
        plt.plot(x, self.predict(x), color="red", label="Model")
        plt.legend()

        plt.show()

    def get_params(self):
        return {"slope": self.w, "bias": self.b}