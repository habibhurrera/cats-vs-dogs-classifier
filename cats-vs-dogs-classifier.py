import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tkinter import Tk, filedialog

# Load trained model
model = load_model("cats_vs_dogs.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(160, 160))  # Match training size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        print(f"Prediction: 🐶 Dog ({prediction[0][0]*100:.2f}%)")
    else:
        print(f"Prediction: 🐱 Cat ({(1 - prediction[0][0])*100:.2f}%)")

if __name__ == "__main__":
    root = Tk()
    root.withdraw()  # Hide main Tkinter window

    while True:
        img_path = filedialog.askopenfilename(
            title="Select an image (Cancel to exit)",
            filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")]
        )

        if not img_path:  # User pressed Cancel
            print("❌ No image selected. Exiting...")
            break

        predict_image(img_path)
        print("-" * 50)  # Separator between predictions
