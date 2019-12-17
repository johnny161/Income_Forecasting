import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 

output_dir = "output/"

def dataProcess_X(rawData):
    if "income" in rawData.columns:
        Data = rawData.drop(['income'], axis=1)
    else:
        Data = rawData
    listObjectColumn = [col for col in Data.columns if Data[col].dtypes == "object"] #读取非数字的column
    listNonObjectColumn = [x for x in list(Data) if x not in listObjectColumn] #读取数字的column
    
    ObjectData = Data[listObjectColumn]
    NonObjectData = Data[listNonObjectColumn]

    ObjectData = pd.get_dummies(ObjectData)
    Data = pd.concat([NonObjectData, ObjectData], axis=1)
    Data.astype("int64")

    Data = (Data - Data.mean(axis=0)) / Data.std(axis=0)
    return Data

def dataProcess_Y(rawData):
    y = rawData['income']
    Data_y = pd.DataFrame((y==' >50K').astype("int64"), columns=["income"])
    return Data_y

def add_layer(inputs, in_size, out_size, activation_function=None):
    W = tf.Variable(tf.random_normal([in_size, out_size], stddev=0.01)) #stddev=1
    b = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    Z = tf.matmul(inputs, W) + b
    if activation_function is None:
        outputs = Z
    else:
        outputs = activation_function(Z)

    return outputs

if __name__ == "__main__":
    trainData = pd.read_csv("../data/adult_train.csv")
    testData = pd.read_csv("../data/x_test.csv")
    ans = pd.read_csv("../data/y_test.csv")

    x_train = dataProcess_X(trainData)
    y_train = dataProcess_Y(trainData).values
    x_test = dataProcess_X(testData)
    y_test = ans['label'].values

    x_train_, x_test_ = x_train.align(x_test, join='left', fill_value=0, axis=1) #类似左外连接同时对齐属性
    x_train = x_train_.values
    x_test = x_test_.values

    # neural network#########################################
    y_train = np.eye(2)[y_train]
    y_test = np.eye(2)[y_test]

    learning_rate = 0.01
    batch_size = 67
    n_epochs = 30

    X = tf.placeholder(tf.float32, [batch_size, 108])
    Y = tf.placeholder(tf.float32, [batch_size, 2])

    # multi-layer############################################
    layer_dims = [108, 15, 2]
    layer_count = len(layer_dims)-1
    layer_iter = X

    for l in range(1, layer_count):
        layer_iter = add_layer(layer_iter, layer_dims[l-1], layer_dims[l], activation_function=tf.nn.relu)
    prediction = add_layer(layer_iter, layer_dims[layer_count-1], layer_dims[layer_count], activation_function=None)
    # multi-layer############################################

    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=prediction)
    loss = tf.reduce_mean(entropy)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        n_examples = int(len(x_train))
        n_batches = int(n_examples/batch_size)
        for i in range(n_epochs):
            ind = [i for i in range(n_examples)]
            np.random.shuffle(ind) # shuffle
            x_train = x_train[ind]
            y_train = y_train[ind]
            #if (i+1) % 5 == 0:
            #    learning_rate = learning_rate * 0.95
            avg_loss = 0.0
            for j in range(n_batches):
                X_batch = x_train[j*batch_size:(j+1)*batch_size]
                Y_batch = y_train[j*batch_size:(j+1)*batch_size].reshape(-1, 2)
                _, _loss = sess.run([optimizer, loss], feed_dict={X:X_batch, Y:Y_batch})
                avg_loss += _loss
                if j == n_batches-1:
                    print("Average Loss of epoch[{0}] lr[{1}]: {2}".format(i, round(learning_rate, 4), (avg_loss/n_batches)))

        # test the model
        n_examples_test = int(len(x_test))
        n_batches = int(n_examples_test/batch_size)
        if(n_examples_test % batch_size != 0): # assert is it divided exactly?
            print("be careful, it cant go through all the test data!")
        total_correct_preds = 0
        for i in range(n_batches):
            X_batch = x_test[i*batch_size:(i+1)*batch_size]
            Y_batch = y_test[i*batch_size:(i+1)*batch_size].reshape(-1, 2)
            preds = sess.run(prediction, feed_dict={X:X_batch})
            correct_preds = tf.equal(tf.argmax(preds,1), tf.argmax(Y_batch,1))
            accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

            total_correct_preds += sess.run(accuracy)
    
        print("the number of test examples is {}".format(n_examples_test))
        print("Accuracy {0}".format(total_correct_preds / float(n_examples_test)))
