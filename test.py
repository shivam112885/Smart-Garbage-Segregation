import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib
import keras.utils as ku
import numpy as np

def predict_image(test_img):
    train = 'Data/Train'

    train_generator = ImageDataGenerator(rescale = 1/255)

    train_generator = train_generator.flow_from_directory(train,
                                                        target_size = (300,300),
                                                        batch_size = 32,
                                                        class_mode = 'sparse')

    labels = (train_generator.class_indices)
    print(labels,'\n')

    labels = dict((v,k) for k,v in labels.items())
    print(labels)

    #load the model
    model = joblib.load('my_model.pkl')

    img = ku.load_img(test_img, target_size = (300,300))
    img = ku.img_to_array(img, dtype=np.uint8)
    img = np.array(img)/255.0
    prediction = model.predict(img[np.newaxis, ...])

    #print("Predicted shape",p.shape)
    print("Probability:",np.max(prediction[0], axis=-1))
    predicted_class = labels[np.argmax(prediction[0], axis=-1)]
    print("Classified:",predicted_class,'\n')


if __name__ == "__main__":
    predict_image('gettyimages-1441946970-612x612.jpg')
