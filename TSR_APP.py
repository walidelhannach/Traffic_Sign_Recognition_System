# Complete project
# Created by: EL HANNACH Walid
# Complete program
# make sure you have all "H5" file of training, testing and "model_trafic.h5" before running the program
# Please change the "general_path" according to your path
# If you have difficulties to run the "TSR_APP.py" you can run the program from the "Main-project.ipynb" file the complete program of the gui is in the last cell 
# For additional information contact me : walid.elhannach@usmba.ac.ma 

import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from sklearn.metrics import accuracy_score
import customtkinter as ctk
from tkinter import filedialog

# Path to your dataset
general_path = r'C:\Users\Dell\Desktop\Road sign recognition Project\DATA\Traffic'

# Fetch Traffic Images
def fetch_traffic_images(traffic_path):
    num_classes = 43
    traffic_images = []
    labels = []
    for label in range(num_classes):
        train_folders = os.path.join(traffic_path, 'Train', str(label))
        imgs = os.listdir(train_folders)
        for img in imgs:
            image = Image.open(train_folders + '\\' + img)
            image = image.resize((30, 30))
            image = np.array(image)
            traffic_images.append(image)
            labels.append(label)
    return traffic_images, labels

# Load Images
X, Y = fetch_traffic_images(general_path)
Y = np.array(Y)
X = np.array(X)

# Load Class Labels
file = open(os.path.join(general_path, 'Classes.txt'))
classes = {}
for line in file:
    data = line.split('-')
    classes.update({data[0]: data[1]})

# Split Data
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
length = len(classes)
train_y = to_categorical(trainY, length)
test_y = to_categorical(testY, length)

# Save Data
pickle.dump(trainX, open(general_path + '\\' + 'trainX.h5', 'wb'))
pickle.dump(testX, open(general_path + '\\' + 'testX.h5', 'wb'))
pickle.dump(trainY, open(general_path + '\\' + 'trainY.h5', 'wb'))
pickle.dump(testY, open(general_path + '\\' + 'testY.h5', 'wb'))

# Model Creation
def createModel(inputShape, outputSize):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', input_shape=inputShape))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(outputSize, activation='softmax'))
    return model

# Compile and Train Model
model = createModel(trainX.shape[1:], length)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(trainX, train_y, epochs=16, validation_data=(testX, test_y), batch_size=32)

# Save Model and History
model.save(general_path + '\\model_traffic.h5')
pickle.dump(hist.history, open(general_path + '\\hist.h5', 'wb'))

# Evaluate Model
value = model.evaluate(testX, test_y)
print('Accuracy: ', round(value[1] * 100, 2), '%')

# Plot Accuracy and Loss
plt.figure(0)
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.figure(1)
plt.plot(hist.history['loss'], label='Training Losses')
plt.plot(hist.history['val_loss'], label='Val Losses')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# GUI Initialization
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
gui = ctk.CTk()
gui.geometry('1200x800')
gui.title('Traffic Sign Recognition || By: EL HANNACH Walid')

# Load Pretrained Model and Data
trainX = pickle.load(open(general_path + '\\trainX.h5', 'rb'))
testX = pickle.load(open(general_path + '\\testX.h5', 'rb'))
trainY = pickle.load(open(general_path + '\\trainY.h5', 'rb'))
testY = pickle.load(open(general_path + '\\testY.h5', 'rb'))
history = pickle.load(open(general_path + '\\hist.h5', 'rb'))
model = load_model(general_path + '\\model_traffic.h5')

# GUI Functions
def classify_image2(path):
    image = Image.open(path)
    image = image.resize((30, 30))
    image = np.array(image)
    plt.imshow(image)
    img = np.expand_dims(image, axis=0)
    vect_output = model.predict(img)
    indx = np.argmax(vect_output)
    definition = classes[str(indx + 1)]
    traffic_class.configure(text=definition)

def addBtnClassify(path):
    btnClassify = ctk.CTkButton(gui, text='Recognize the Road Sign', command=lambda: classify_image2(path), width=250, height=50, font=('Arial', 16, 'bold'))
    btnClassify.place(relx=0.40, rely=0.7)

def uploadImage():
    file = filedialog.askopenfilename()
    default = Image.open(file)
    default = default.resize((360, 360))
    default = ImageTk.PhotoImage(default)
    traffic_image.configure(image=default)
    traffic_image.image = default
    addBtnClassify(file)

# GUI Elements
btnUpload = ctk.CTkButton(gui, command=uploadImage, text="Upload a Traffic Sign Image", width=250, height=50, font=('Arial', 16, 'bold'))
traffic_image = ctk.CTkLabel(gui, text="")
traffic_class = ctk.CTkLabel(gui, text="", text_color='white', font=('Arial', 18, 'bold'))

# Display Default Image
default = Image.open(r'C:\Users\Dell\Desktop\Road sign recognition Project\DATA\Traffic\img.jpg')
default = default.resize((360, 360))
default = ImageTk.PhotoImage(default)
traffic_image.configure(image=default)
traffic_image.image = default

# Add Creator Information
info_frame = ctk.CTkFrame(gui)
info_frame.place(relx=0.05, rely=0.02, anchor='nw')
created_by_label = ctk.CTkLabel(info_frame, text="Developed by: EL HANNACH Walid", text_color='white', font=('Arial', 14, 'bold'))
created_by_label.pack()

# Pack GUI Elements
btnUpload.pack(pady=50)
traffic_image.pack()
traffic_class.pack(pady=50)

# Run GUI Main Loop
gui.mainloop()

