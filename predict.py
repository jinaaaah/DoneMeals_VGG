from keras.preprocessing import image
from keras.models import Model, Sequential, load_model
from keras.applications.vgg19 import preprocess_input
import numpy as np

model = load_model(filepath='./model/model2.hdf5')

img_path = './data/images/test/img2.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

print(features)