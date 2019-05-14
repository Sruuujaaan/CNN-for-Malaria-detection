from keras.models import load_model
import pickle
classifier = load_model('Malaria.h5')


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('A:/Kaggle Datasets/Malaria Detection/training_set/Uninfected/C1_thinF_IMG_20150604_104722_cell_9.png', target_size = (80, 80))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'NotInfected'
else:
    prediction = 'Infected'

print(prediction)