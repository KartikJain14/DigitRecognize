import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load dataset (using iris for demonstration)
data = load_iris()
X, y = data.data, data.target

# Train a Random Forest model
model = RandomForestClassifier()
model.fit(X, y)

# Get feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importance
plt.figure()
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), np.array(data.feature_names)[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.savefig('feature_importance.png')
plt.show()
