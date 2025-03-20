from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from datetime import datetime


model = load_model("gen_model.h5")
dest = "generation_results"
num_images = 20
latent_dim = 100
noise = np.random.normal(0, 1, (num_images, latent_dim))
image_list = []


try:
    print("generating images")
    images = model.predict(noise)

    #for debugging
    #print(f"Generated image shape: {images.shape}\n{images[0, 0, 0]}")  

    images = ((images + 1) * 127.5).astype(np.uint8)
except Exception as e:
    print(f"Error with image generation: {e}")
    


print("saving images")

try:
    for i in images:
        curr_date = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        img_bgr = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
        file_path = os.path.join(dest, f"image_{curr_date}.jpg")
        success = cv2.imwrite(file_path, img_bgr)
        if not success:
            print(f"Failed to save {file_path}")
except Exception as e:
    print(f"Error with saving images: {e}")