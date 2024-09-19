import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from keras.models import load_model
from keras.datasets import mnist

# Load your model
model = load_model('model.keras')

# Load or prepare your test dataset
(x_test, y_test), _ = mnist.load_data()  # Example dataset
x_test = x_test.astype('float32') / 255.0  # Normalize if needed
x_test = np.expand_dims(x_test, axis=-1)  # Add channel dimension if needed

# Make predictions
y_pred_probs = model.predict(x_test)

# If it's multi-class, convert to binary for one class (e.g., class 1)
class_index = 1  # Change this to the class you want to analyze
y_test_binary = (y_test == class_index).astype(int)
y_pred_probs_binary = y_pred_probs[:, class_index]  # Probability for the specific class

# Compute Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test_binary, y_pred_probs_binary)

# Plot Precision-Recall curve
plt.figure()
plt.plot(recall, precision, color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('precision_recall_curve.png')
plt.show()
