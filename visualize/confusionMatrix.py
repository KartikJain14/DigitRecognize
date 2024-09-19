import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import load_model
from keras.datasets import mnist  # Example dataset

# Load your model
model = load_model('model.keras')

# Load or prepare your test dataset
(x_test, y_test), _ = mnist.load_data()  # Use your test dataset here
x_test = x_test.astype('float32') / 255.0  # Normalize if needed
x_test = np.expand_dims(x_test, axis=-1)  # Add channel dimension if needed

# Make predictions
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class labels

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y_test), 
            yticklabels=np.unique(y_test))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

# Save the figure as an image
plt.savefig('confusion_matrix.png')  # You can also use other formats like .jpg, .pdf, etc.

# Optional: Display with sklearn's ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')

# Save the sklearn confusion matrix display
plt.savefig('confusion_matrix_sklearn.png')  # Save this visualization as well

plt.show()

