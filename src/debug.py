from tensorflow.keras.layers import Input
from anfis_model import *
from anfis_layers import *
import tensorflow as tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D  # pycharm auto import

# reset console, just in case
tf.keras.backend.clear_session()

np.random.seed(1)
np.set_printoptions(precision=3, suppress=True)


def func(x, y):
    return (math.sin(6*x)+math.cos(y))/4+0.5


parameters_count = 2
parameters_sets_count = [3, 4]
parameters_sets_total_count = sum(parameters_sets_count)

input_data_size = 1024
test_data_size = 32

debug = False

# ------------
# LAYERS & DEBUG
# ------------

f0 = Input(shape=(parameters_count,))
f1 = FuzzificationLayer(fuzzy_sets_count=parameters_sets_count)(f0)
f2 = RulesLayer(fuzzy_sets_count=parameters_sets_count)(f1)
f3 = SumNormalisationLayer()(f2)
f4 = DefuzzificationLayer()([f0, f3])

anfis = tf.keras.Model(inputs=f0, outputs=f4)
forward = tf.keras.Model(inputs=f0, outputs=f3)

dummy = None
if debug:
    dummy = [
        tf.keras.Model(inputs=f0, outputs=f0),
        tf.keras.Model(inputs=f0, outputs=f1),
        tf.keras.Model(inputs=f0, outputs=f2),
        tf.keras.Model(inputs=f0, outputs=f3),
        tf.keras.Model(inputs=f0, outputs=f4),
    ]


# ------------
# TRAINING
# ------------

din = []
dout = []

import math
for i in range(input_data_size):
    x = list(np.random.uniform(0.0, 1.0, size=2))
    din.append(x)
    y = func(x[0], x[1])
    dout.append([y])

data_input = tf.convert_to_tensor(din, dtype='float32')
data_output = tf.convert_to_tensor(dout, dtype='float32')

anfis.compile(loss=tf.metrics.MAE,
              optimizer=tf.keras.optimizers.SGD(
                  clipnorm=0.5,
                  learning_rate=1e-4),
              metrics=[tf.keras.metrics.RootMeanSquaredError()]
              )

train_anfis(anfis, forward, data_input, data_output,
            epochs=10, batch_size=32,
            learning_rate=1-1e-3,
            dummy=dummy)

dpd = anfis.predict(data_input)

# ------------
# TESTING
# ------------

tin = []
tout = []

for i in range(test_data_size):
    x = list(np.random.uniform(0.0, 1.0, size=2))
    tin.append(x)
    y = func(x[0], x[1])
    tout.append([y])

test_input = tf.convert_to_tensor(tin, dtype='float32')
test_output = tf.convert_to_tensor(tout, dtype='float32')

tpd = anfis.predict(test_input)

# ------------
# PLOTTING - TRAINING/TESTING
# ------------

x_d = np.array(din)[:, 0]
x_t = np.array(tin)[:, 0]

y_d = np.array(din)[:, 1]
y_t = np.array(tin)[:, 1]

z_do, z_dp = np.array(dout), np.array(dpd)
z_to, z_tp = np.array(tout), np.array(tpd)

fig = plt.figure(figsize=(12, 6))
ax = [
    fig.add_subplot(1, 2, 1, projection='3d'),
    fig.add_subplot(1, 2, 2, projection='3d')
    ]

ax[0].scatter(x_d, y_d, z_do)
ax[0].scatter(x_d, y_d, z_dp)

ax[1].scatter(x_t, y_t, z_to)
ax[1].scatter(x_t, y_t, z_tp)

for i in range(2):
    ax[i].set_xlabel('x')
    ax[i].set_ylabel('y')
    ax[i].set_zlabel('z')

# ------------
# PLOTTING - MEMBERSHIP PARAMETERS FUNCTION
# ------------

mf_a = anfis.trainable_variables[0].numpy()
mf_b = anfis.trainable_variables[1].numpy()
mf_c = anfis.trainable_variables[2].numpy()
mf_x = np.linspace(0.0, 1.0, 1000)

mf_rows = math.floor(math.sqrt(parameters_count))
mf_cols = math.ceil(parameters_count / mf_rows)

fig, ax = plt.subplots(nrows=mf_rows, ncols=mf_cols, figsize=(12, 6))
ax = np.reshape(ax, (mf_rows, mf_cols))

t = 0
j = parameters_sets_count[t]
for i in range(parameters_sets_total_count):
    if i >= j:
        t += 1
        j += parameters_sets_count[t]
    ax[t // mf_cols][t % mf_cols].plot(mf_x, 1.0 / (1.0 + (((mf_x - mf_c[i]) / mf_a[i]) ** 2) ** (1.0 / mf_b[i])))

plt.show()

print("END")
