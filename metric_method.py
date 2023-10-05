import numpy as np

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Рассчитываем расстояния между x и всеми точками в обучающем наборе
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # Получаем индексы k ближайших точек
        k_indices = np.argsort(distances)[:self.k]
        # Извлекаем метки классов для k ближайших точек
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Возвращаем наиболее часто встречающийся класс среди k ближайших соседей
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

# Пример использования
if __name__ == "__main__":
    # Создаем небольшой обучающий набор данных
    X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
    y_train = np.array([0, 0, 1, 1])

    # Создаем экземпляр классификатора KNN
    knn = KNNClassifier(k=2)

    # Обучаем модель на обучающем наборе
    knn.fit(X_train, y_train)

    # Создаем тестовый набор данных
    X_test = np.array([[2, 2], [5, 5]])

    # Делаем прогнозы
    predictions = knn.predict(X_test)

    print("Прогнозы:", predictions)
