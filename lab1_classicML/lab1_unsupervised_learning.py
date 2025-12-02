import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1. Генерація даних
# Використовується фіксоване random_state=0 для відтворюваності результатів
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 2. Реалізація власного класифікатора K-Means
class CustomKMeans:
    def __init__(self, n_clusters=4, max_iter=300, random_state=42):
        self.k = n_clusters
        self.max_iter = max_iter
        self.history = []
        self.random_state = random_state  # Фіксація початкової випадковості

    def fit(self, X):
        # Ініціалізація генератора псевдовипадкових чисел для стабільності результатів
        np.random.seed(self.random_state)
        
        # Вибір початкових центрів кластерів випадковим чином
        idx = np.random.choice(len(X), self.k, replace=False)
        centroids = X[idx]
        
        # Збереження початкового стану
        self.history.append((centroids.copy(), np.zeros(len(X))))

        for i in range(self.max_iter):
            # Крок 1: Обчислення відстаней між вибірками та поточними центроїдами
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Крок 2: Оновлення центроїдів як середнього значення точок кластера
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.k)])

            # Збереження поточного стану для подальшої візуалізації
            self.history.append((new_centroids.copy(), labels.copy()))

            # Перевірка критерію збіжності
            if np.all(centroids == new_centroids):
                print(f"In-house зійшовся за {i} ітерацій")
                break

            centroids = new_centroids
        
        self.centroids = centroids
        self.labels = labels

print("Запуск In-house...")
my_model = CustomKMeans(n_clusters=4, random_state=42)
my_model.fit(X)

print("Запуск Sklearn...")
sklearn_model = KMeans(n_clusters=4, init='random', n_init=10, max_iter=300, random_state=0)
sklearn_model.fit(X)

# 3. Візуалізація результатів кластеризації
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plt.subplots_adjust(bottom=0.25)

def update_plot(frame_idx):
    axes[0].clear()
    centroids, labels = my_model.history[frame_idx]
    
    # Візуалізація поточного стану кластеризації: точки та центроїди
    axes[0].scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', alpha=0.6)
    axes[0].scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', edgecolors='black')
    axes[0].set_title(f"In-house алгоритм: ітерація {frame_idx}")
    axes[0].grid(True, alpha=0.3)

# Налаштування слайдера для перемикання між ітераціями
ax_slider = plt.axes([0.15, 0.1, 0.35, 0.03])
slider = Slider(ax_slider, 'Ітерація', 0, len(my_model.history) - 1, 
                valinit=len(my_model.history) - 1, valstep=1)

def on_change(val):
    update_plot(int(slider.val))
    fig.canvas.draw_idle()

slider.on_changed(on_change)

# Відображення результатів кластеризації алгоритмом sklearn
axes[1].scatter(X[:, 0], X[:, 1], c=sklearn_model.labels_, s=50, cmap='viridis')
centers = sklearn_model.cluster_centers_
axes[1].scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', edgecolors='black')
axes[1].set_title("Результат кластеризації (Sklearn)")
axes[1].grid(True, alpha=0.3)

# Ініціалізація візуалізації
update_plot(len(my_model.history) - 1)
import os
if os.environ.get('MPLBACKEND') == 'Agg':
    # Якщо ми в Docker - зберігаємо картинку
    print("Running in Docker environment. Saving image to 'output.png'...")
    # Малюємо останній кадр для збереження
    update_plot(len(my_model.history) - 1)
    plt.savefig("output_docker.png")
    print("Done! Image saved.")
else:
    # Якщо ми локально - показуємо вікно
    plt.show()
