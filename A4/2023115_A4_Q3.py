import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
np.random.seed(2023115)

class_0_A = np.random.multivariate_normal(mean=[-3,-3], cov=[[1,0],[0,1]], size=200)
class_1_A = np.random.multivariate_normal(mean=[3,3], cov=[[1,0],[0,1]], size=200)

class_0_B = np.random.multivariate_normal(mean=[-3,-3], cov=[[3,0],[0,3]], size=200)
class_1_B = np.random.multivariate_normal(mean=[3,3], cov=[[3,0],[0,3]], size=200)

datasetA = np.vstack((class_0_A, class_1_A))
labelsA = np.hstack((np.zeros(200), np.ones(200)))
datasetB = np.vstack((class_0_B, class_1_B))
labelsB = np.hstack((np.zeros(200), np.ones(200)))

x_train_A, x_test_A, y_train_A, y_test_A = train_test_split(datasetA, labelsA, test_size=0.3, random_state=2023115)
x_train_B, x_test_B, y_train_B, y_test_B = train_test_split(datasetB, labelsB, test_size=0.3, random_state=2023115)

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
    def fit(self, X, y, max_iters=1000):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        misclassifications_per_epoch = []
        convergence_epoch = None
        
        for epoch in range(max_iters):
            errors = 0
            for i in range(n_samples):
                linear_output = np.dot(self.w, X[i]) + self.b
                y_predicted = 1 if linear_output >= 0 else 0
                update = y[i] - y_predicted
                self.w += self.learning_rate * update * X[i]
                self.b += self.learning_rate * update
                errors += int(update != 0)
            
            misclassifications_per_epoch.append(errors)

            if (errors == 0):
                convergence_epoch = epoch
                break
        
        return misclassifications_per_epoch, convergence_epoch
    
    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        y_predicted = np.where(linear_output >= 0, 1, 0)
        return y_predicted


def plot_decision_boundary(model, X_train, y_train, X_test, y_test, title, dataset_name):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # Training data plot
    axes[0].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    axes[0].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='red', label='Class 0', s=30)
    axes[0].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='blue', label='Class 1', s=30)
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].set_title(f'{title} - Training Data')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    axes[1].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='red', label='Class 0', s=30)
    axes[1].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='blue', label='Class 1', s=30)
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].set_title(f'{title} - Validation Data')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'decision_boundary_{dataset_name}.png', dpi=100, bbox_inches='tight')
    plt.show()


def plot_misclassifications(misclass_list, convergence_epoch, dataset_name):
    plt.figure(figsize=(10, 6))
    epochs = range(len(misclass_list))
    plt.plot(epochs, misclass_list, marker='o', linestyle='-', linewidth=2, markersize=4)
    if convergence_epoch is not None:
        plt.axvline(x=convergence_epoch, color='r', linestyle='--', label=f'Convergence at Epoch {convergence_epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Misclassified Samples')
    plt.title(f'Misclassifications per Epoch - Dataset {dataset_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'misclassifications_{dataset_name}.png', dpi=100, bbox_inches='tight')
    plt.show()


print("\n")
print("DATASET A (Covariance = I)")
print("\n")
perceptron_A = Perceptron(learning_rate=0.01)
misclass_A, convergence_A = perceptron_A.fit(x_train_A, y_train_A, max_iters=300)
y_pred_A = perceptron_A.predict(x_test_A)
accuracy_A = np.mean(y_pred_A == y_test_A)
print(f"Convergence Epoch: {convergence_A if convergence_A is not None else 'Did not converge within 300 epochs'}")
print(f"Test Accuracy: {accuracy_A:.4f}")
plot_misclassifications(misclass_A, convergence_A, 'A')
plot_decision_boundary(perceptron_A, x_train_A, y_train_A, x_test_A, y_test_A, 'Dataset A (Covariance = I)', 'A')



print("\n")
print("DATASET B (Covariance = 3I)")
print("\n")
perceptron_B = Perceptron(learning_rate=0.01)
misclass_B, convergence_B = perceptron_B.fit(x_train_B, y_train_B, max_iters=300)
y_pred_B = perceptron_B.predict(x_test_B)
accuracy_B = np.mean(y_pred_B == y_test_B)
print(f"Convergence Epoch: {convergence_B if convergence_B is not None else 'Did not converge within 300 epochs'}")
print(f"Test Accuracy: {accuracy_B:.4f}")
plot_misclassifications(misclass_B, convergence_B, 'B')
plot_decision_boundary(perceptron_B, x_train_B, y_train_B, x_test_B, y_test_B,'Dataset B (Covariance = 3I)','B')
