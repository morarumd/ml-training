import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split

num_classes = 10
digits = datasets.load_digits()
X, y = digits.images, digits.target
x_mean = X.mean(axis=0)
X = X - x_mean
y = y.reshape(y.size, -1)
x_train, x_test, y_train, y_test = train_test_split(X, y)

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.models.Sequential()
model.add(Flatten())
model.add(Dense(25, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(20, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform'))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4),
              metrics=['accuracy'])

fit_history = model.fit(x_train, y_train, validation_data=(x_test, y_test), 
                        batch_size=32, epochs=500, verbose=2)

print('Test:', model.evaluate(x_test, y_test, verbose=0))
print('Train:', model.evaluate(x_train, y_train, verbose=0))

plt.plot(fit_history.history['loss'], label='train_loss')
plt.plot(fit_history.history['val_loss'], label='test_loss')
plt.ylim((0,6))
plt.legend()
plt.savefig('mnist_keras.png')