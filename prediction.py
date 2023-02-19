import xlrd
import xlwt
import tensorflow as tf
from tensorflow.metrics import mean_squared_error
import random
#from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

def save_dataset(filename, prediction, realvalue):
    '''
    save the prediction as xls file
    '''
    book = xlwt.Workbook()
    sheet1 = book.add_sheet("Sheet1")
    for num in range(len(prediction)):
        row = sheet1.row(num)
        row.write(0, prediction[num])
        row.write(1, realvalue[num])
    book.save(filename)


def load_dataset(filename):
    '''
    load dataset.xls and convert it into list
    '''
    with xlrd.open_workbook(filename, 'rb') as dataset:
        table = dataset.sheets()[0]
        nrows = table.nrows
        ncols = table.ncols
        datas = []
        for row in range(1, nrows):
            row_data = [table.cell(row, 0).value]  # the first col saves date
            for col in range(1, ncols):
                # save as float type
                row_data.append(float(table.cell(row, col).value))
            datas.append(row_data)
    return datas


def reshape_datas(datas):
    '''
    raw:datas.shape=(nrows-1,ncols)
    target: shaped_data.shape=(one_hot,)
    temperature | sunshine | air pressure | wind speed | rainfall
    380           20         40             10           20      = 470
    76            5          5              5            5       = 96
    '''
    reshape_datas = []
    for row in range(len(datas)):  # nrows-1
        # temperature 380: [-2,36)
        temp_dewtemp = [0] * 310
        #print(temp_temp)
        temp_indice = int(datas[row][1]*10)
        temp_dewtemp[temp_indice] = 1
        # sunshine 20: [0,20)
        temp_pressure = [0] * 30
        temp_indice = int(round(datas[row][2]/1000))
        temp_pressure[temp_indice] = 1
        # air pressure 40: [0,40)
        temp_maxtemp = [0] * 40
        temp_indice = int(round(datas[row][3]))
        temp_maxtemp[temp_indice] = 1
        
        temp_mintemp = [0] * 40
        temp_indice = int(round(datas[row][4]))
        temp_mintemp[temp_indice] = 1
        
                # wind speed 10: [0,10)
        temp_humidity= [0] * 10
        temp_indice = int(round(datas[row][6]/100))
        temp_humidity[temp_indice] = 1
        # rainfall 20: [0,200)
        temp_wd = [0] * 20
        temp_indice = int(round(datas[row][7]))
        temp_wd[temp_indice] = 1
        
        temp_ws = [0] * 50
        temp_indice = int(round(datas[row][8]))
        temp_ws[temp_indice] = 1
        
        temp_rainfall = [0] * 40
        temp_indice = int(round(datas[row][9] / 100.0))
        temp_rainfall[temp_indice] = 1
        # concate all temps
        row_data = temp_dewtemp + temp_pressure + temp_maxtemp + temp_mintemp+ temp_humidity + temp_wd + temp_ws+temp_rainfall
        reshape_datas.append(row_data)
    print(reshape_datas)
    return reshape_datas
 
#    reshape_datas = []
#    for row in range(len(datas)):  # nrows-1
#        # temperature 380: [-2,36)
#        temp_dewtemp = [0] * 310
#        #print(temp_temp)
#        temp_indice = int(datas[row][1]*10)
#        temp_dewtemp[temp_indice] = 1
#        # sunshine 20: [0,20)
#        temp_tmin = [0] * 40
#        temp_indice = int(round(datas[row][4]))
#        temp_tmin[temp_indice] = 1
#        # air pressure 40: [0,40)
#        temp_y = [0] * 10
#        temp_indice = int(round(datas[row][10]/100000))
#        temp_y[temp_indice] = 1
#        
#                # wind speed 10: [0,10)
#        temp_d= [0] * 10
#        temp_indice = int(round(datas[row][11]))
#        temp_d[temp_indice] = 1
#        # rainfall 20: [0,200)
#        temp_cosy = [0] * 200
#        temp_indice = int(datas[row][12]*10)+10
#        temp_cosy[temp_indice] = 1
#        
#        temp_cosd = [0] * 10
#        temp_indice = int(round(datas[row][13]))
#        temp_cosd[temp_indice] = 1
#  
#        # concate all temps
#        row_data = temp_dewtemp + temp_tmin + temp_y + temp_d + temp_cosy + temp_cosd
#        reshape_datas.append(row_data)
##   print(reshape_datas) 
#    return reshape_datas

def get_random_batch(datas, n_steps, batch_size):
    '''
    input in the form of (temp,sun,air,wind,rain) shape=(5,)
    output in the form of (0,0,...,1,...,0) shape=(380,)
    [-2,36) 1 decimal points, (36-(-2))*10
    '''
    random_batch_x = []
    random_batch_y = []
    indices = np.random.randint(0, len(datas) - n_steps, batch_size)
    for i in indices:
        # random_batch_x.shape=(batch_size,n_steps,470)
        temp_x = []  # temp_x.shape=(n_steps,470)
        for step in range(n_steps):
            temp_x.append(datas[i + step])  # datas[i].shape=(470,)
        random_batch_x.append(temp_x)
        temp_y = datas[i + n_steps][:310]
        random_batch_y.append(temp_y)
    return random_batch_x, random_batch_y


def get_all_batch(datas, n_steps):
    batch_x = []
    batch_y = []
    for indice in range(len(datas) - n_steps):
        temp_x = []
        for step in range(n_steps):
            temp_x.append(datas[indice + step])
        batch_x.append(temp_x)
        temp_y = datas[indice + n_steps][:310]
        batch_y.append(temp_y)
    return batch_x, batch_y


def reform_y_display(test_y):
    # test_y.shape=(len(test_datas),380)
    indices = np.argmax(test_y, 1)  # indices.shape=(len(test_datas),)
    display_y = (indices) / 10.0
    return display_y

lstm_size=2 #two is the best
training_iters = 35000
learning_rate =  0.001
display_step = 200
batch_size = 32
# network parameters'
n_input=540# combination2
#n_input = 580# combination1
n_steps = 1
n_hidden = 60
n_output = 310

# tf graph weights, biases
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_output])
# define weights, biases
weights = {
    'out': tf.Variable(tf.truncated_normal([n_hidden, n_output]))
}
biases = {
    'out': tf.Variable(tf.truncated_normal([n_output]))
}
def lstm_cell(size):
    return [tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0) for _ in range(size)]

def RNN(x, weights, biases):
    # permuting batch_size and n_input
    x = tf.transpose(x, [1, 0, 2])
    # reshape into (n_steo*batch_size,n_input)
    x = tf.reshape(x, [-1, n_input])
    # split to get a list of 'n_steps'
    #x = tf.split(0, n_steps, x)
    x = tf.split(x, n_steps, 0)
    
    with tf.variable_scope('n_steps4'):
        #lstm_cell = rnn_cell.BasicLSTMCell(
#        lstm_cell = rnn.BasicLSTMCell(
#            n_hidden, forget_bias=1.0)
        stacked_lstm=tf.contrib.rnn.MultiRNNCell(lstm_cell(lstm_size),state_is_tuple=True)
        #outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
        outputs, states = rnn.static_rnn(stacked_lstm, x, dtype=tf.float32)
#    return tf.matmul(outputs[-1], weights['out']) + biases['out']
        w1=tf.layers.dense(outputs[-1],units=n_hidden)
        w2=tf.layers.dense(w1,units=n_hidden,activation=tf.nn.relu)
        return tf.matmul(w2,weights['out'])+biases['out']

pred = RNN(x, weights, biases)
# define loss and optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
## Create hyperparameter space
#epochs = [5, 10]
#batches = [5, 10, 100]
#optimizers = ['rmsprop', 'adam']
#hyperparameters = dict(optimizer=optimizers, epochs=epochs, batch_size=batches)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
# evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# read dataset
filename = 'sittwe_2015-2016-clean.xlsx'
datas = load_dataset(filename)
datas = reshape_datas(datas)
print('Data Reading Finished!')

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = get_random_batch(datas, n_steps, batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter" + str(step * batch_size) + ", Minibatch Loss=" +
                  "{:.6f}".format(loss) + ", Training Accuracy=" +
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # test
    test_datas = load_dataset('sittwee_2017_clean.xlsx')
    test_datas = reshape_datas(test_datas)
    print("Test Data Reading Finished!")
    test_x, test_y = get_all_batch(test_datas, n_steps)
    pred_test_y = sess.run(pred, feed_dict={x: test_x})
    test_step = 0
    display_pred_test_y = reform_y_display(pred_test_y)
    display_test_y = reform_y_display(test_y)
    while test_step < len(test_datas) - n_steps:
        if test_step % 3 == 0:
            print("pred: {}, real: {}".format(
                display_pred_test_y[test_step], display_test_y[test_step]
            ))
        test_step += 1
    print("Testing Cost:",
          sess.run(cost, feed_dict={x: test_x, y: test_y}))
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={x: test_x, y: test_y}))
#    print("Testing Accuracy:",
#          sess.run(R_square,feed_dict={x: test_x, y: test_y}))
#    print("RMSE:",
#          sess.run(RMSE,feed_dict={x: test_x, y: test_y}))
    
    R2 = r2_score(display_test_y,display_pred_test_y, multioutput='variance_weighted')
    print(R2)
    MSE=mean_squared_error(display_test_y,display_pred_test_y)
    print(MSE)



# change the n_steps and name of the file
save_dataset("2017_2.xlsx", display_pred_test_y, display_test_y)
print("Prediction Saved")

plt.plot(display_pred_test_y, 'g')
plt.plot(display_test_y, 'r')
red_patch = mpatches.Patch(color='red', label='real')
green_patch = mpatches.Patch(color='green', label='pred')
plt.legend(handles=[red_patch, green_patch])
plt.show()
