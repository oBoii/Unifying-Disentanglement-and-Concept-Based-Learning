# loads d-sprites dataset and creates a subset of it
import numpy as np

file = "./dsprites-dataset/" + "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
dataset = np.load(file, encoding='bytes', allow_pickle=True)
images = dataset['imgs']
latents_values = dataset['latents_values']
latents_classes = dataset['latents_classes']
metadata = dataset['metadata'][()]

# 10% of the dataset
n = len(images)
n_subset = int(0.1 * n)
indices = np.random.choice(n, n_subset, replace=False)
images_subset = images[indices]
latents_values_subset = latents_values[indices]
latents_classes_subset = latents_classes[indices]

np.savez_compressed("./dsprites-dataset/dsprites_subset.npz", imgs=images_subset, latents_values=latents_values_subset,
                    latents_classes=latents_classes_subset, metadata=metadata)


