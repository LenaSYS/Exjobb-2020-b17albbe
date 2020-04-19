from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import time


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

base_model = VGG16 (weights='imagenet', include_top=False)

# load the model
model = VGG16(weights='imagenet', include_top=False)

start = time.time()

# load an image from file
image = load_img('cat.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model.predict(image)
# convert the probabilities to class labels
label = decode_predictions(yhat)

print(label)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]

end = time.time()
# print(end - start)

confusion = tf.math.confusion_matrix(labels=[0, 9], predictions=[0, 0])

figure = plt.figure(figsize=(8, 8))
sns.heatmap(confusion, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# print the classification
print('%s (%.2f%%)' % (label[1], label[2] * 100))
print(confusion)
print(label)
