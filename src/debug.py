from tensorflow.keras.layers import Input
from anfis_model import *
from anfis_layers import *
import tensorflow as tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D  # pycharm auto import

# reset console, just in case
tf.keras.backend.clear_session()

np.random.seed(0)


def func(x, y):
    return (math.sin(6*x)+math.cos(y))/4+0.5


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
for i in range(256):
    x = list(np.random.uniform(0.0, 1.0, size=2))
    din.append(x)
    y = func(x[0], x[1])
    dout.append([y])

data_input = tf.convert_to_tensor(din, dtype='float32')
data_output = tf.convert_to_tensor(dout, dtype='float32')

#anfis.compile(loss=root_mean_squared_error, optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])

# train_anfis(anfis, forward, data_input, data_output, epochs=100, batch_size=3,
#             loss=tf.keras.losses.MAE,
#             optimizer=tf.keras.optimizers.SGD(clipnorm=1e-3)
#             )

anfis.compile(loss=tf.metrics.MAE,
              optimizer=tf.keras.optimizers.SGD(
                  clipnorm=0.5,
                  learning_rate=1e-4),
              metrics=["accuracy"])
train_anfis(anfis, forward, data_input, data_output,
            epochs=10, batch_size=1,
            learning_rate=1e-2,
            dummy=[
                tf.keras.Model(inputs=f0, outputs=f0),
                tf.keras.Model(inputs=f0, outputs=f1),
                tf.keras.Model(inputs=f0, outputs=f2),
                tf.keras.Model(inputs=f0, outputs=f3),
                tf.keras.Model(inputs=f0, outputs=f4),
            ]
            )
#
# tin = []
# tout = []
#
# for i in range(30):
#     x = list(np.random.uniform(0.0, 1.0, size=2))
#     tin.append(x)
#     y = func(x[0], x[1])
#     tout.append([y])
#
# test_input = tf.convert_to_tensor(tin, dtype='float32')
# test_output = tf.convert_to_tensor(tout, dtype='float32')

#data_predict = anfis(data_input)
#test_predict = anfis(test_input)

dpd = anfis.predict(data_input)#data_predict.numpy()
#rpd = test_predict.numpy()
#rout = test_output.numpy()

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

ax = plt.axes()
aa = anfis.trainable_variables[0].numpy()
bb = anfis.trainable_variables[1].numpy()
cc = anfis.trainable_variables[2].numpy()
xx = np.linspace(0.0, 1.0, 1000)

t = 0
j = parameters_sets_count[t]
for i in range(aa.shape[0]):
    if i >= j:
        t += 1
        j += parameters_sets_count[t]
        plt.show()
        ax = plt.axes()
    ax.plot(xx, 1.0 / (1.0 + (((xx - cc[i]) / aa[i]) ** 2) ** (1.0 / bb[i])))

plt.show()

print("END")
