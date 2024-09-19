import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.datasets import mnist  # Example dataset, adjust as needed

# Load your trained model
model = load_model('model.keras')

# Load or prepare your test dataset
(x_test, y_test), _ = mnist.load_data()  # Example dataset
x_test = x_test.astype('float32') / 255.0  # Normalize if needed
x_test = np.expand_dims(x_test, axis=-1)  # Add channel dimension if needed

# Make predictions
y_pred_probs = model.predict(x_test)

# Plot histogram for each class
num_classes = y_pred_probs.shape[1]  # Assuming y_pred_probs is (num_samples, num_classes)

plt.figure(figsize=(14, 7))  # Increase figure size for better readability

for i in range(num_classes):
    plt.hist(y_pred_probs[:, i], bins=30, alpha=0.5, label=f'Class {i}', edgecolor='black')

plt.title('Histogram of Prediction Probabilities for Each Class', fontsize=16)
plt.xlabel('Prediction Probability', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)  # Adjust x-tick labels font size
plt.yticks(fontsize=12)  # Adjust y-tick labels font size
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.75)  # Add gridlines for better visibility
plt.tight_layout()  # Adjust layout to make it fit better
plt.savefig('histogram_predictions.png')
plt.show()
