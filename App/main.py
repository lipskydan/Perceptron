import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
      Скорость обучения (между 0.0 и 1.0)
    n_iter : int
      Passes over the training dataset.
      Проходы по обучающему набору данных.
    random_state : int
      Random number generator seed for random weight initialization.
      Начальное значение генератора случайных чисел для инициализации случайными весами.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
      Веса после подгонки.
    errors_ : list
      Number of misclassifications (updates) in each epoch.
      Количество неправильных классификаций (обновлений) в каждой эпохе.

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples (количество образцов) and
          n_features is the number of features (количество признаков).
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


v1 = np.array([1, 2, 3])
v2 = 0.5 * v1
np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# ## Training a perceptron model on the Iris dataset

# ...

# ### Reading-in the Iris data

df = df = pd.read_csv('/Users/danial/Desktop/Perceptron/iris.data', header=None)
df.tail()

df = pd.read_csv('iris.data', header=None)
df.tail()

# select setosa (щетинистый) and versicolor (разноцветный)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='щетинистый')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='разноцветный')

plt.xlabel('длина чашелистика [cм]')
plt.ylabel('длина лепестка [см]')
plt.legend(loc='upper left')
plt.title('График рассеяния')

plt.savefig('images/scatter-plot.png', dpi=300)

plt.show()


# ### Training the perceptron model

ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Количество обновлений')
plt.title('График ошибок неправильной классификации')

plt.savefig('images/misclassification-errors-plot.png', dpi=300)

plt.show()
