import numpy as np
import matplotlib.pyplot as plt


class GeneralizedLinearRegressionGD_Hazem:
    """
    Linear / Multiple / Polynomial Regression using Gradient Descent
    Supports:
    - Normal Linear Regression
    - Polynomial Regression (via feature transformation)
    - Ridge (L2) Regularization
    - Lasso (L1) Regularization
    """

    def __init__(self, alpha=0.01, num_iterations=100):
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.w = None
        self.b = 0
        self.lambda_ = 0.0
        self.reg_type = "none"   # "none", "ridge", "lasso"
        self.MSE_values = []

    # ---------- Feature Engineering ----------

    def polynomial_transform(self, x, degree):
        """
        Transforms single feature x into polynomial features:
        x -> [x, x^2, ..., x^degree]
        """
        return np.column_stack([x ** d for d in range(1, degree + 1)])

    # ---------- Core ML Methods ----------

    def initialize_parameters(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def compute_gradients(self, X, y, y_hat):
        n = len(y)

        dw = (1 / n) * np.dot(X.T, (y_hat - y))
        db = (1 / n) * np.sum(y_hat - y)

        # Regularization
        if self.reg_type == "ridge":        # L2
            dw += self.lambda_ * self.w

        elif self.reg_type == "lasso":      # L1
            dw += self.lambda_ * np.sign(self.w)

        return dw, db

    def compute_mse(self, y, y_hat):
        return np.mean((y_hat - y) ** 2)

    # ---------- Training ----------

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.initialize_parameters(n_features)
        self.MSE_values = []

        for _ in range(self.num_iterations):
            y_hat = self.predict(X)
            dw, db = self.compute_gradients(X, y, y_hat)

            self.w -= self.alpha * dw
            self.b -= self.alpha * db

            y_hat = self.predict(X)
            self.MSE_values.append(self.compute_mse(y, y_hat))

        return self

    # ---------- Explicit APIs (BEST DESIGN) ----------

    def fit_linear(self, X, y):
        self.reg_type = "none"
        return self.fit(X, y)

    def fit_ridge(self, X, y, lambda_):
        self.lambda_ = lambda_
        self.reg_type = "ridge"
        return self.fit(X, y)

    def fit_lasso(self, X, y, lambda_):
        self.lambda_ = lambda_
        self.reg_type = "lasso"
        return self.fit(X, y)

    def fit_polynomial(self, x, y, degree):
        X_poly = self.polynomial_transform(x, degree)
        self.reg_type = "none"
        return self.fit(X_poly, y)

    def fit_polynomial_ridge(self, x, y, degree, lambda_):
        X_poly = self.polynomial_transform(x, degree)
        self.lambda_ = lambda_
        self.reg_type = "ridge"
        return self.fit(X_poly, y)

    def fit_polynomial_lasso(self, x, y, degree, lambda_):
        X_poly = self.polynomial_transform(x, degree)
        self.lambda_ = lambda_
        self.reg_type = "lasso"
        return self.fit(X_poly, y)

    # ---------- Utilities ----------

    def plot_loss(self):
        plt.plot(self.MSE_values)
        plt.xlabel("Iteration")
        plt.ylabel("MSE")
        plt.title("MSE over Iterations")
        plt.show()

    def get_params(self):
        return {
            "weights": self.w,
            "bias": self.b
        }