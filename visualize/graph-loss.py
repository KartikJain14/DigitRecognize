import matplotlib.pyplot as plt
from keras.models import load_model

# Load your model and training history
model = load_model('model.keras')

# Assuming you have saved training history
# For example: history = model.fit(...) 
# Load your history data here (if saved in a file) or create history manually for demonstration
# Example of manual history for demonstration purposes
history = {
    'loss': [0.5, 0.3, 0.2, 0.15],
    'val_loss': [0.6, 0.4, 0.35, 0.3],
    'accuracy': [0.7, 0.8, 0.85, 0.9],
    'val_accuracy': [0.65, 0.75, 0.8, 0.85]
}

# Plotting Loss
plt.figure()
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')
plt.show()

# Plotting Accuracy
plt.figure()
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_curve.png')
plt.show()
