# Load the d-sprite dataset and store each record as a separate file. Images are saved as "dsprites-dataset/input/{i}.png"
# and latents are saved as "dsprites-dataset/latents/{i}.npy"

import numpy as np
import cv2
import os

file = "./dsprites-dataset/" + "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
dataset = np.load(file, encoding='bytes', allow_pickle=True)
images = dataset['imgs']
latents_values = dataset['latents_values']
latents_classes = dataset['latents_classes']
metadata = dataset['metadata'][()]


def save_data():
    # create dir if not exists
    os.makedirs("./dsprites-dataset/input", exist_ok=True)
    os.makedirs("./dsprites-dataset/latent_vals", exist_ok=True)
    os.makedirs("./dsprites-dataset/latent_classes", exist_ok=True)

    for i in range(len(images)):
        # save images
        img = images[i]
        img = img * 255
        img = img.astype(np.uint8)
        img = img.squeeze()  # from (1, 64, 64) to (64, 64)
        file = f"./dsprites-dataset/input/{i}.png"
        cv2.imwrite(file, img)

        # save latents
        latent = latents_values[i]
        file = f"./dsprites-dataset/latent_vals/{i}.npy"
        np.save(file, latent)

        # save latents
        latent = latents_classes[i]
        file = f"./dsprites-dataset/latent_classes/{i}.npy"
        np.save(file, latent)

        # logging
        if i % 1000 == 0:
            print(f"Saved {i} images of {len(images)}")

save_data()
