import numpy as np
import random
import cv2
import os

from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

from utils import *

# 关闭tf warning
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#数据目录，jpg格式
img_path = './Minist Data/trainingSample/trainingSample'

#得到数据列表
image_paths = list(paths.list_images(img_path))

#读取数据
image_list, label_list = load(image_paths, verbose=100)

#onehot化标签
lb = LabelBinarizer()
label_list = lb.fit_transform(label_list)

#分割数据集,9:1的train/test
X_train, X_test, y_train, y_test = train_test_split(image_list, 
                                                    label_list, 
                                                    test_size=0.1, 
                                                    random_state=42)

#创建客户端
clients_batched = create_clients(X_train, y_train, num_clients=10, initial='client')

#制造test batch
test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))

comms_round = 100

#create optimizer
lr = 0.01 
loss='categorical_crossentropy'
# 度量
metrics = ['accuracy']
# 优化器选择SGD
optimizer = SGD(lr=lr, 
                decay=lr / comms_round, 
                momentum=0.9
               ) 

#初始化模型
# 使用MNIST，shape参数将是28*28*1 = 784
smlp_global = SimpleMLP()
global_model = smlp_global.build(shape=784, class_num=10)

print('start federated learning!')

# 外循环是全局模型，内循环是本地模型
for comm_round in range(comms_round):

    # 获得全局模型的初始化权值
    global_weights = global_model.get_weights()

    #储存客户端训练模型后的参数
    scaled_local_weight_list = list()

    #随机化客户端的字典顺序，使得训练随机
    client_names= list(clients_batched.keys())
    random.shuffle(client_names)

    #对于每个客户端初始化一个与当前全局模型一样权重的模型
    for client in client_names:
        smlp_local = SimpleMLP()
        local_model = smlp_local.build(shape=784, class_num=10)
        local_model.compile(loss=loss, 
                      optimizer=optimizer, 
                      metrics=metrics)

        #把本地权重设置为和全局模型一样
        local_model.set_weights(global_weights)

        #使用本地数据进行训练，进行一轮epoch
        local_model.fit(clients_batched[client], epochs=1, verbose=0)

        #训练后将权重缩放并附加到scaled_local_weight_list中
        scaling_factor = weight_scalling_factor(clients_batched, client)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)

        #清除内存
        K.clear_session()

    #将所有客户端的比例权重相加
    average_weights = sum_scaled_weights(scaled_local_weight_list)

    #使用这个更新模型
    global_model.set_weights(average_weights)

    #使用预留的测试集对全局模型进行测试
    for(X_test, Y_test) in test_batched:
        global_acc,global_loss = test_model(X_test, Y_test, global_model, comm_round)

SGD_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(320)
smlp_SGD = SimpleMLP()
SGD_model = smlp_SGD.build(784, 10)

SGD_model.compile(loss=loss, 
              optimizer=optimizer, 
              metrics=metrics)

print('start SGD model!')
# 训练SGD模型
_ = SGD_model.fit(SGD_dataset, epochs=100, verbose=0)

#test the SGD global model and print out metrics
for(X_test, Y_test) in test_batched:
        SGD_acc, SGD_loss = test_model(X_test, Y_test, SGD_model, 1)