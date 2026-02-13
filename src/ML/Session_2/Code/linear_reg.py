import numpy as np
import matplotlib.pyplot as plt

class LinearRegression_Hazem:

    def __init__(self, alpha=0.01, num_iterations=100):
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.w = 0
        self.b = 0
        self.SSE_values = []
        self.MSE_values = []

    def predict(self, x):
        return self.w * x + self.b

    def compute_gradients(self, x, y, y_hat):
        dw = np.mean((y_hat - y) * x)
        db = np.mean(y_hat - y)
        return dw, db

    def compute_sse(self, y, y_hat):
        return np.sum((y_hat - y) ** 2)
    
    
    def compute_mse(self, y, y_hat):
        return np.mean((y_hat - y) ** 2)


    def fit(self, x, y):
        for i in range(self.num_iterations):
            y_hat = self.predict(x)

            dw, db = self.compute_gradients(x, y, y_hat)

            self.w -= self.alpha * dw
            self.b -= self.alpha * db

            #re-compute y_hat
            y_hat = self.predict(x)

            sse = self.compute_sse(y, y_hat)
            self.SSE_values.append(sse)

            mse = self.compute_mse(y, y_hat)
            self.MSE_values.append(mse)

            if (i + 1) % 20 == 0:
                print(f"Iteration {i+1}, SSE: {sse} , MSE: {mse} ")

        return self


    def plot(self, x, y):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(self.num_iterations), self.SSE_values, label="SSE")
        plt.xlabel("Iteration")
        plt.ylabel("SSE")
        plt.title("SSE over Iterations")
        plt.legend()


        plt.subplot(1, 2, 2)
        plt.scatter(x, y, color="blue", label="Data Points")
        plt.plot(x, self.predict(x), color="red", label="Regression Line")
        plt.legend()
        plt.show()

    def get_params(self):
        return {
            "slope": self.w,
            "bias": self.b
        }
