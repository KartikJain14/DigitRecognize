import keras
import visualkeras
from keras.models import load_model

# Load the Keras model
model = load_model('model.keras')

# Create a more descriptive visualization
visualization = visualkeras.layered_view(
    model,
    to_file='model_visualization.png',  # Output file name
    legend=True,                         # Show legend for layer types
    color_map={                        # Custom colors for layer types
        'Dense': 'blue',
        'Conv2D': 'orange',
        'MaxPooling2D': 'green',
        'Dropout': 'red',
        'InputLayer': 'purple',
    }
)

print("Model visualization saved as 'model_visualization.png'")
