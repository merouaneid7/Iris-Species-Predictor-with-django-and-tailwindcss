from django.shortcuts import render
from .models import IrisData, Prediction
from sklearn.datasets import load_iris
import numpy as np
import tensorflow as tf

iris = load_iris()
X = iris.data
y = iris.target

class KNN:
    def __init__(self, X_train, y_train, n_neighbors=5):
        self.X_train = X_train
        self.y_train = y_train
        self.n_neighbors = n_neighbors

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.argmax(np.bincount(k_nearest_labels))
        return most_common

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.learning_rate * y_[idx]

    def predict(self, X):
        linear_model = np.dot(X, self.weights) - self.bias
        return np.sign(linear_model)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, n_iterations=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_input_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output
        return self.output

    def backward(self, X, y, output):
        error = y - output
        d_output = error * self.sigmoid_derivative(output)
        error_hidden_layer = d_output.dot(self.weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(self.hidden_output)
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * self.learning_rate
        self.bias_hidden_output += np.sum(d_output) * self.learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden_layer) * self.learning_rate
        self.bias_input_hidden += np.sum(d_hidden_layer) * self.learning_rate


def cnn_predict(sepal_length, sepal_width, petal_length, petal_width):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    prediction = model.predict(np.array([[sepal_length, sepal_width, petal_length, petal_width]]))
    predicted_class = iris.target_names[np.argmax(prediction)]
    return predicted_class

def ann_predict(sepal_length, sepal_width, petal_length, petal_width):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    prediction = model.predict(np.array([[sepal_length, sepal_width, petal_length, petal_width]]))
    predicted_class = iris.target_names[np.argmax(prediction)]
    return predicted_class

def home(request):
    return render(request, 'home.html')

def predict(request):
    if request.method == 'POST':
        sepal_length = float(request.POST['sepal_length'])
        sepal_width = float(request.POST['sepal_width'])
        petal_length = float(request.POST['petal_length'])
        petal_width = float(request.POST['petal_width'])
        selected_model = request.POST['selected_model']

        if selected_model == 'KNN':
            model = KNN(X, y)
            predicted_class = iris.target_names[model.predict([sepal_length, sepal_width, petal_length, petal_width])]
        elif selected_model == 'Logistic Regression':
            model = LogisticRegression()
            model.fit(X, y)
            predicted_class = iris.target_names[model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]]
        elif selected_model == 'SVM':
            model = SVM()
            model.fit(X, y)
            predicted_class = iris.target_names[model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]]
        elif selected_model == 'Neural Network':
            model = NeuralNetwork(input_size=4, hidden_size=5, output_size=3)
            model.forward([[sepal_length, sepal_width, petal_length, petal_width]])
            predicted_class = iris.target_names[np.argmax(model.output)]
        elif selected_model == 'CNN':
            predicted_class = cnn_predict(sepal_length, sepal_width, petal_length, petal_width)
        elif selected_model == 'ANN':
            predicted_class = ann_predict(sepal_length, sepal_width, petal_length, petal_width)


          
        Prediction.objects.create(
            model_used=selected_model,
            predicted_class=predicted_class,
            
            
        )

        return render(request, 'prediction_result.html', {'predicted_class': predicted_class    })

    return render(request, 'predict.html')
