"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Toprak Kis, Jasper Schommartz
         
08/2022
"""


# %%
"""
Import modules

"""

import models as lm
import data as ld
from matplotlib import pyplot as plt
import tensorflow as tf
import datetime
now = datetime.datetime.now

# %% Own modules


# %%
"""
Load model

"""

model = lm.main()


# %%
"""
Load data

"""

xs, ys, xs_c, ys_c = ld.bathtub()

# %%
"""
Model calibration

"""

t1 = now()
print(t1)

tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
h = model.fit([xs_c], [ys_c], epochs=5000,  verbose=2)

t2 = now()
print('it took', t2 - t1, '(sec) to calibrate the model')
print(lm._x_to_y())

# plot some results
plt.figure(1, dpi=600)
plt.semilogy(h.history['loss'], label='training loss')
plt.grid(which='both')
plt.xlabel('calibration epoch')
plt.ylabel('log$_{10}$ MSE')
plt.legend()


# %%
"""
Evaluation

"""

plt.figure(2, dpi=600)
plt.scatter(xs_c[::10], ys_c[::10], c='green', label='calibration data')
plt.plot(xs, ys, c='black', linestyle='--', label='bathtub function')
plt.plot(xs, model.predict(xs), label='model', color='red')
plt.title('Softplus: layers = 2, nodes = 16, epochs = 5000')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
