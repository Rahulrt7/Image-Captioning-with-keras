import numpy as np
import copy
import h5py
import matplotlib.pyplot as plt
from utils.coco_utils import load_coco_data
from IPython.display import Image, display
get_ipython().magic(u'matplotlib inline')

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, TimeDistributed, Merge
from keras.layers import Embedding, Dropout
from keras.layers import LSTM
from keras.layers.core import RepeatVector
from keras import callbacks, utils, optimizers


# Loading the data to Memory
data = load_coco_data(pca_features=True)

# Print out all the keys and values from the data dictionary
for k, v in data.iteritems():
  if type(v) == np.ndarray:
    print k, type(v), v.shape, v.dtype
  else:
    print k, type(v), len(v)

train_captions_mat = data['train_captions']
train_image_idxs = data['train_image_idxs']
train_features = data['train_features']
val_captions_mat = data['val_captions']
val_image_idxs = data['val_image_idxs']
val_features = data['val_features']
idx_to_word = data['idx_to_word']

a=  data['val_image_idxs'][3]
b = data['val_urls'][a]

print train_captions_mat[0]
for i in train_captions_mat[0]:
    print idx_to_word[i],


# Number of images to use for training and validation

limit1 = len(train_image_idxs)
limit2 = len(val_image_idxs)
num_train_img = limit1
num_val_img = limit2
num_train_cap = limit1
num_val_cap = limit2


# Constructng image_train and image_val

image_train = np.zeros((num_train_img, 512))
image_val = np.zeros((num_val_img, 512))
for i in range(num_train_img):
    index = train_image_idxs[i]
    image_train[i] = train_features[index]
    
for i in range(num_val_img):
    index = val_image_idxs[i]
    image_val[i] = val_features[index]

print image_train.shape, image_val.shape


# Construct word_train and word_val

word_train = train_captions_mat[:num_train_cap]
word_val = val_captions_mat[:num_val_cap]
print word_train.shape, word_val.shape


# Sparse encoding

print train_captions_mat[0]
print train_captions_mat.shape

y_train = np.zeros((num_train_img, 17, 1))
y_val = np.zeros((num_val_img, 17, 1))
y_train.shape

for i in range(num_train_img):
    for j in range(17):
        y_train[i][j] = train_captions_mat[i][j]
        
for i in range(num_val_img):
    for j in range(17):
        y_val[i][j] = val_captions_mat[i][j]

print y_train[1000]
print train_captions_mat[1000]

print y_train.shape, y_val.shape

print len(y_train[0][0])


# Feeding input into the model to fit sentence by sentence in a LOOP

# create the model
embedding_vector_length = 512
vocabulary_size = 1004
max_caption_length = 17
batch_size = 64          # one parameter update per sentence
image_vector_length = 512

# image vector
image_model = Sequential()
image_model.add(Dense(image_vector_length, input_dim=512))
image_model.add(RepeatVector(max_caption_length))

# caption vector
word_model = Sequential()
word_model.add(Embedding(input_dim=vocabulary_size, 
                    output_dim=embedding_vector_length, input_length=17))

# Merge models

model = Sequential()
model.add(Merge([image_model, word_model], mode='concat'))  # merging layers
model.add(LSTM(512, unroll=True, return_sequences=True, implementation=2, stateful=False))
model.add(Dropout(rate=0.5))
model.add(TimeDistributed(Dense(vocabulary_size)))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, momentum=0.0, nesterov=False)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.summary()
# path to checkpoints
filepath = './checkpoints/weights-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, verbose=0, mode='max')
board = callbacks.TensorBoard(log_dir='./tensorboard_logs', histogram_freq=0, write_graph=True, write_images=False)


model_history = model.fit([image_train, word_train], y_train, verbose=2, callbacks=[checkpoint, board], 
                validation_data=([image_val, word_val], y_val) , batch_size=batch_size, shuffle=True, epochs=100)

score = model.evaluate([image_val, word_val], y_val, verbose=2, batch_size=batch_size)

index = 0
prediction = model.predict_classes([image_val[0:100], word_val[0:100]], verbose=2)[index]
print prediction


# Probability of top four words at each position(pos=17)

prob = model.predict_proba([image_val[:500], word_val[:500]], verbose=2)[index]
print prob.shape
prob.sort()
for i in range(17):
    print prob[i][1000:]

actual = val_captions_mat[index]

def arr_to_sent(predicted, actual):
    s = []
    m = data['idx_to_word']
    for i in predicted:
        s.append(m[i])
    print " ".join(s)
    s = []
    for i in actual:
        s.append(m[i])
    print " ".join(s)

arr_to_sent(prediction, actual)

a = data['val_image_idxs'][index]
print a
b = data['val_urls'][a]
display(Image(b))


# Displaying all captions

def array_to_sentence(arr):
    s = []
    m = data['idx_to_word']
    for i in arr: 
        s.append(m[i])
    return " ".join(s)

def generate_all_captions(a):
    test = []
    for i, v in enumerate(data['val_image_idxs']):
        if v == a :
            print array_to_sentence(val_captions_mat[i])
            test.append(i)
            
    print test 

generate_all_captions(a)


