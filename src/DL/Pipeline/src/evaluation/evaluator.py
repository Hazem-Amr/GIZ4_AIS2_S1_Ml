class Evaluator:

    def evaluate(self, model, x_test, y_test):
        loss, acc = model.evaluate(x_test, y_test)
        return {"loss": loss, "accuracy": acc}