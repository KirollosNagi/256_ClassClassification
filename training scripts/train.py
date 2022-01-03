from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import joblib
import numpy as np

# Load data set
x_train = joblib.load("x_train.dat")
y_train = joblib.load("y_train.dat")
x_test = joblib.load("x_test.dat")
y_test = joblib.load("y_test.dat")

# display data set shape
print('x_train shape:', {x_train.shape})
print('y_train shape:', {y_train.shape})
print('x_test shape:', {x_test.shape})
print('y_test shape:', {y_test.shape})

# Create a model and add layers
model = Sequential()

model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(257, activation='softmax'))

model.summary()

# Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

# Train the model
model.fit(
    x_train,
    y_train,
    epochs=40,
    batch_size=512,
    verbose=2,
    shuffle=True
)

# Save neural network structure
model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)

# Save neural network's trained weights
model.save_weights("model_weights.h5")

# Evaluate test set
score = model.evaluate(x_test, y_test, verbose=0)
#print(score)
print('test score', score[0])
print('test accuracy', score[1])

# Try a predict as well to manually calculate the ACCR5
results = model.predict(x_test)

top1 = 0.0
top5 = 0.0

for i, l in enumerate(y_test):
    class_prob = results[i]
    top_values = (-class_prob).argsort()[:5]
    '''print('l: ', l)
    print('class_prob: ', class_prob)
    print('top_values: ', top_values)
    print('top_values[0]: ', top_values[0])'''
    if top_values[0] == np.argmax(l):
        top1 += 1.0
    if np.isin(np.array([np.argmax(l)]), top_values):
        top5 += 1.0

print("top1 acc:", top1/len(y_test))
print("top5 acc:", top5/len(y_test))