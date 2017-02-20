# coding:utf-8
from keras.layers import Dense, Dropout, SimpleRNN, GaussianNoise,Activation,Embedding,Input,GRU, LSTM,\
    merge, Convolution1D, Flatten, MaxPooling1D
from keras.models import Sequential,Model
from keras.utils import np_utils
from keras.regularizers import l1l2, l2
from keras.optimizers import RMSprop,SGD,adagrad, rmsprop,adadelta
from knn_or_svm import data_x, data_y, dictionary_most2000
from my_util import get_hand_operated_features, record_error_distribution, error_distribution_path_for_lstm, read_and_compute_error_distribution
from sklearn.cross_validation import train_test_split
from PIL import Image
from os import listdir
import numpy as np
import sys, getopt, os

nDim_vector = 200
MAX_LENGTH = 120
EMBEDDING_PATH_SERVER = "/home/ht/glove.6B/glove.6B.200d.txt"
EMBEDDING_PATH_HOME = "E:\\glove.6B\\glove.6B.100d.txt"
EMBEDDING_PATH = ""

data_y = data_y[:, 0]
opts, args = getopt.getopt(sys.argv[1:], "m:")
if opts[0][1] == "S":
    print "Run on the Server now."
    EMBEDDING_PATH = EMBEDDING_PATH_SERVER
elif opts[0][1] == "H":
    print "Run on the Home now."
    EMBEDDING_PATH = EMBEDDING_PATH_HOME
else:
    print "input error"
    os._exit(0)

'''
def my_rnn(data_x, data_y):
    model = Sequential()
    # model.add(GaussianNoise(0.1, input_shape=(28, 28)))
    model.add(SimpleRNN(output_dim=128, activation='relu', input_shape=(28, 28)))
    model.add(Dense(50))
    model.add(Activation("relu"))
    model.add(Dense(10))
    model.add(Activation("softmax"))
    my_rmsprop = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=my_rmsprop, metrics=["accuracy"])
    model.fit(data_x, data_y)


def load_data():
    data = np.empty(shape=(42000, 1, 28, 28), dtype="float32")
    label = np.empty(shape=(42000, 1), dtype="uint8")
    imgs = listdir("C:\\Users\\lenovo-gjfht-real\\Desktop\\mnist\\mnist")
    print len(imgs)
    print imgs[1]

    for i in range(42000):
        img = Image.open("C:\\Users\\lenovo-gjfht-real\\Desktop\\mnist\\mnist\\" + imgs[i])
        arr = np.asarray(img, dtype="float32")
        data[i, 0, :, :] = arr
        label[i] = int(imgs[i][0])
    return data, label

data, label = load_data()
label = np_utils.to_categorical(label, 10)
data = data.reshape((42000, 28, 28))
data /= 255


'''

# first step load data 
my_dictionary = {}
index2word = {}
current_index = 1  
for line in data_x:
    for word in line:
        if word not in my_dictionary.keys():
            my_dictionary[word] = current_index
            index2word[current_index] = word
            current_index += 1
print current_index
print my_dictionary["happen"]


embedding_index = {}
f = open(EMBEDDING_PATH, 'rb')
for line in f:
    values = line.split()
    word = values[0]
    emb = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = emb
f.close()
print "get %d word vectors" % len(embedding_index)


zero_embedding = np.zeros((nDim_vector,), dtype="float32")
embedding_matrix = np.zeros((len(my_dictionary)+1, nDim_vector), dtype="float32")
for word in my_dictionary.keys():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[my_dictionary[word]] = embedding_vector
print "get the embedding matrix"
print embedding_matrix[100]


x_train = np.zeros((len(data_x), MAX_LENGTH), dtype="int32")
for i in range(len(data_x)):
    line = data_x[i]
    length = min(MAX_LENGTH, len(line))
    for j in range(length):
        x_train[i][j] = my_dictionary[line[j]]
print x_train[109]


my_embedding_layer = Embedding(len(my_dictionary)+1,
                               nDim_vector,
                               weights=[embedding_matrix],
                               mask_zero=False,          
                               input_length=MAX_LENGTH,
                               trainable=True)
my_embedding_layer_for_cnn = Embedding(len(my_dictionary)+1,
                               nDim_vector,
                               weights=[embedding_matrix],
                               mask_zero=False,          
                               input_length=MAX_LENGTH,
                               trainable=True)

# bulid model
my_seq_inputs = Input(shape=(MAX_LENGTH,), dtype="int32", name="seq_inputs")
my_length_inputs = Input(shape=(3, ), dtype="float32", name="length_inputs")
embedded_sequence = my_embedding_layer(my_seq_inputs)
embedded_sequence_for_cnn = my_embedding_layer_for_cnn(my_seq_inputs)
x1_ = GRU(128, activation='relu',return_sequences=True, W_regularizer=l2(0.01))(embedded_sequence)
x1 = GRU(50, activation='relu', return_sequences=False)(x1_)     
x2_ = GRU(128, activation='relu',return_sequences=True, go_backwards=True, W_regularizer=l2(0.01))(embedded_sequence)
x2 = GRU(50, activation='relu', return_sequences=False, go_backwards=False)(x2_)
x = merge([x1, x2], mode='concat') 

# embedded_sequence_for_cnn  = Dropout(0.3)(embedded_sequence_for_cnn)
# embedded_sequence = Dropout(0.3)(embedded_sequence) 
x3_1 = Convolution1D(nb_filter=32, filter_length=3, init='uniform',
                   border_mode='same', input_length=MAX_LENGTH)(embedded_sequence)
x3_2 = Convolution1D(nb_filter=32, filter_length=2, init='uniform',
					border_mode='same', input_length=MAX_LENGTH)(embedded_sequence_for_cnn)
x3_3 = Convolution1D(nb_filter=32, filter_length=1, init='uniform',
					border_mode='same', input_length=MAX_LENGTH)(embedded_sequence_for_cnn)
x4_1 = Convolution1D(nb_filter=32, filter_length=3, init='uniform',
                    border_mode='same', input_length=MAX_LENGTH)(x1_)
x4_2 = Convolution1D(nb_filter=32, filter_length=3, init='uniform',
                    border_mode='same', input_length=MAX_LENGTH)(x2_)
cnn_feature = Flatten()(x3_1)
cnn_feature = Dense(100, activation='relu')(cnn_feature)
x = merge([x, cnn_feature], mode='concat')
x = Dropout(0.7)(x)
# x = Dense(100, activation='relu')(x)
x = Dense(12, activation='relu', W_regularizer=l1l2(l2=0.01))(x)
cnn_feature = Dropout(0.3)(cnn_feature)
x = merge([cnn_feature, my_length_inputs], mode="concat")             
my_outputs = Dense(4, activation='softmax')(x)
my_model = Model(input=[my_seq_inputs, my_length_inputs], output=my_outputs)
# my_rmsprop = rmsprop(lr=0.0008, rho=0.94)
my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# training
y_train = np_utils.to_categorical(data_y, nb_classes=4)
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_train, y_train, test_size=0.1)
x_train2 = get_hand_operated_features(x_train1, index2word, dictionary_most2000)
x_test2 = get_hand_operated_features(x_test1, index2word, dictionary_most2000)
CLASS_WETGHT = {0:0.9, 1:1.3, 2:1.3, 3:0.9 }
my_model.fit([x_train1, x_train2], y_train1, validation_data=([x_test1, x_test2], y_test1), nb_epoch=50, batch_size=128, shuffle=True)
Predict = my_model.predict([x_test1, x_test2], batch_size = 2046, verbose=1)
Pred_Label = np.argmax(Predict, axis=1)
Pred_Label = np.asarray(Pred_Label)
y_test1 = np.argmax(y_test1, axis=1)
y_test1 = np.asarray(y_test1)
print y_test1.shape
print Pred_Label.shape
print np.mean(y_test1 == Pred_Label)
print y_test1 == Pred_Label


record_error_distribution(y_test1, Pred_Label==y_test1, error_distribution_path_for_lstm)
read_and_compute_error_distribution(error_distribution_path_for_lstm)
