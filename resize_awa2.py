# for every image in awa2-dataset/AwA2-data/Animals_with_Attributes2/<class_name>/<img_name>.jpg
#     resize the image to 64x64 and move it to awa2-dataset/AwA2-data/Animals_with_Attributes2_resized/<class_name>/<img_name>.jpg

import os
from PIL import Image

# Path to the directory containing the dataset
path = './awa2-dataset/AwA2-data/Animals_with_Attributes2/JPEGImages'
path_resized = './awa2-dataset/AwA2-data/Animals_with_Attributes2_resized/JPEGImages'

# Create the directory if it does not exist
if not os.path.exists(path_resized):
    os.makedirs(path_resized)

# Iterate over the classes
for class_name in os.listdir(path):
    if os.path.isdir(os.path.join(path, class_name)):
        print(class_name)
        if not os.path.exists(os.path.join(path_resized, class_name)):
            os.makedirs(os.path.join(path_resized, class_name))
        # Iterate over the images
        for img_name in os.listdir(os.path.join(path, class_name)):
            if img_name.endswith('.jpg'):
                img = Image.open(os.path.join(path, class_name, img_name))
                img = img.resize((64, 64))
                img.save(os.path.join(path_resized, class_name, img_name))
                img.close()
print('Done')

