import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from keras.models import load_model
from keras.datasets import mnist  # Example dataset

# Load your model
model = load_model('model.keras')

# Load or prepare your test dataset
(x_test, y_test), _ = mnist.load_data()  # Use your actual test dataset
x_test = x_test.astype('float32') / 255.0  # Normalize if needed
x_test = np.expand_dims(x_test, axis=-1)  # Add channel dimension if needed

# Make predictions
y_pred_probs = model.predict(x_test)

# If it's multi-class, convert to binary for one class (e.g., class 1)
# Modify this section based on your specific use case
class_index = 1  # Change this to the class you want to analyze
y_test_binary = (y_test == class_index).astype(int)
y_pred_probs_binary = y_pred_probs[:, class_index]  # Probability for the specific class

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_probs_binary)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.show()
