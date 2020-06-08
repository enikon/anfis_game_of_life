from tensorflow.keras.layers import Input
from anfis_model import *
from anfis_layers import *
import tensorflow as tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D  # pycharm auto import

# reset console, just in case
tf.keras.backend.clear_session()

parameters_count = 2
parameters_sets_count = [3, 4]

f0 = Input(shape=(parameters_count,))
f1 = FuzzificationLayer(fuzzy_sets_count=parameters_sets_count)(f0)
f2 = RulesLayer(fuzzy_sets_count=parameters_sets_count)(f1)
f3 = SumNormalisationLayer()(f2)
f4 = DefuzzificationLayer()([f0, f3])

anfis = tf.keras.Model(inputs=f0, outputs=f4)
forward = tf.keras.Model(inputs=f0, outputs=f3)

din = []
dout = []

import math
for i in range(120):
    x = list(np.random.uniform(0.0, 1.0, size=2))
    din.append(x)
    y = ((math.cos(5.0*x[0]+3.0*x[1])+1.0)/2.0+x[0]*x[0]/7.0-x[1]*x[1]/3.0)
    dout.append([y])

data_input = tf.convert_to_tensor(din, dtype='float32')
data_output = tf.convert_to_tensor(dout, dtype='float32')

#anfis.compile(loss=root_mean_squared_error, optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])

# train_anfis(anfis, forward, data_input, data_output, epochs=100, batch_size=3,
#             loss=tf.keras.losses.MAE,
#             optimizer=tf.keras.optimizers.SGD(clipnorm=1e-3)
#             )

anfis.compile(loss=tf.metrics.MAE, optimizer=tf.keras.optimizers.SGD(clipnorm=1.00), metrics=["accuracy"])
train_anfis(anfis, forward, data_input, data_output, epochs=20, batch_size=1, loss=tf.metrics.MAE)

tin = []
tout = []

for i in range(30):
    x = list(np.random.uniform(0.0, 1.0, size=2))
    tin.append(x)
    y = ((math.cos(5.0*x[0]+3.0*x[1])+1.0)/2.0+x[0]*x[0]/7.0-x[1]*x[1]/3.0)
    tout.append([y])

test_input = tf.convert_to_tensor(tin, dtype='float32')
test_output = tf.convert_to_tensor(tout, dtype='float32')

data_predict = anfis(data_input)
test_predict = anfis(test_input)

dpd = data_predict.numpy()
rpd = test_predict.numpy()
rout = test_output.numpy()

x = np.array(din)[:, 0]
y = np.array(din)[:, 1]
z_d = np.array(dout)
z_p = np.array(dpd)

ax = plt.axes(projection='3d')
ax.scatter(x, y, z_d)
ax.scatter(x, y, z_p)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
