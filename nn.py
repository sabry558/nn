import tkinter as tk
import customtkinter as ctk
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from models import models  # Assumes your model class is in models.py

def button_clicked():
    features = ['gender', 'body mass', 'beak_length', 'beak_depth', 'fin_length']
    classes = ['A', 'B', 'C']

    # Retrieve values from the GUI fields
    epochs = epochs_entry.get()
    mse = mse_entry.get()
    lr = lr_entry.get()
    f1 = features_box1.get()
    f2 = features_box2.get()
    c1 = class_box1.get()
    c2 = class_box2.get()

    # Validation checks
    if f1 == f2:
        error_label.configure(text="You can't choose the same feature twice")
        return
    elif c1 == c2:
        error_label.configure(text="You can't choose the same class twice")
        return
    elif not epochs or not epochs.isdigit():
        error_label.configure(text="Please enter a valid epochs number")
        return
    try:
        mse_value = float(mse)
    except ValueError:
        error_label.configure(text="Please enter a valid float for MSE threshold")
        return
    try:
        lr_value = float(lr)
    except ValueError:
        error_label.configure(text="Please enter a valid float for Learning rate")
        return
    if model_choice.get() == "":
        error_label.configure(text="Please choose a model")
        return
    
    error_label.configure(text="")


    model = models(learning_rate=lr_value, mse_threshold=mse_value, epochs=int(epochs), bias=bias_check.get() == 1)
    model.read_csv("birds.csv")
    
    if model_choice.get() == "Perceptron":
        predictions, X_test,Y_test = model.preceptron_model(features.index(f1), features.index(f2), c1, c2)
    elif model_choice.get() == "Adaline":
        predictions, X_test,Y_test = model.adaline_model(features.index(f1), features.index(f2), c1, c2)

    plot_decision_boundary(X_test, Y_test, model)

def plot_decision_boundary(X_test, Y_test, model):
    ax.clear()  
    m=0
    if model.bias:
        m=1
    for i, label in enumerate(np.unique(Y_test)):
        ax.scatter(X_test[Y_test == label][:, m], X_test[Y_test == label][:, m+1], 
                   label=f'Class {label}', edgecolor='k', s=50)

    x_min, x_max = X_test[:, m].min() - 1, X_test[:, m].max() + 1
    y_min, y_max = X_test[:, m+1].min() - 1, X_test[:, m+1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    if model.bias:
        grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
    else:
        grid = np.c_[xx.ravel(), yy.ravel()]

    Z = np.array([model.predict_perceptron(g) for g in grid]) if model_choice.get() == "Perceptron" else np.array([model.predict_adaline(g) for g in grid])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.2)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    ax.set_title("Decision Boundary")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    canvas.draw()  

root = tk.Tk()
root.configure(bg="#2E2E2E")
root.geometry("1280x720")

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1A1A1A'
plt.rcParams['axes.facecolor'] = '#1A1A1A'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'

fig, ax = plt.subplots()
fig.set_size_inches(6, 5)
canvas = FigureCanvasTkAgg(master=root, figure=fig)
plot = canvas.get_tk_widget()
plot.place(x=500, y=10)

features_box1 = ctk.CTkComboBox(root, values=['gender', 'body mass', 'beak_length', 'beak_depth', 'fin_length'])
features_box2 = ctk.CTkComboBox(root, values=['gender', 'body mass', 'beak_length', 'beak_depth', 'fin_length'])
features_box1.place(x=10, y=30)
features_box2.place(x=10, y=70)

class_box1 = ctk.CTkComboBox(root, values=['A', 'B', 'C'])
class_box2 = ctk.CTkComboBox(root, values=['A', 'B', 'C'])
class_box1.place(x=10, y=110)
class_box2.place(x=10, y=150)

epochs_entry = ctk.CTkEntry(root, placeholder_text='epochs')
epochs_entry.place(x=10, y=190)
mse_entry = ctk.CTkEntry(root, placeholder_text='mse threshold')
mse_entry.place(x=10, y=230)
lr_entry = ctk.CTkEntry(root, placeholder_text='learning rate')
lr_entry.place(x=10, y=270)

model_choice = tk.StringVar(value="")
perceptron_radio = ctk.CTkRadioButton(root, text="Perceptron", variable=model_choice, value="Perceptron")
perceptron_radio.place(x=10, y=310)
adaline_radio = ctk.CTkRadioButton(root, text="Adaline", variable=model_choice, value="Adaline")
adaline_radio.place(x=110, y=310)

bias_check = ctk.CTkCheckBox(root, text='Bias')
bias_check.place(x=10, y=350)

error_label = ctk.CTkLabel(root, text_color="red", text="")
error_label.place(x=10, y=380)
predict_button = ctk.CTkButton(root, text="Predict", command=button_clicked)
predict_button.place(x=10, y=410)

root.mainloop()
