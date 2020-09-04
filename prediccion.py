
import cv2
import numpy as np
from keras.models import load_model
#
# modelo = './modelo/modelo.h5'
# pesos = './modelo/pesos.h5'
#
# model = load_model(modelo)
# model.load_weights(pesos)
#
# imagen = cv2.imread('navaja-de-acero-inoxidable-y-madera-noble-85cm.jpg')
# origen = imagen.copy()
#
# imagen = cv2.resize(imagen, (100, 100))
# imagen = imagen.astype('float')/255.0
# imagen = np.expand_dims(imagen, axis=0)
#
# (no_knife, knife) = model.predict(imagen)[0]
#
# if(no_knife > knife):
#     print("El modelo asegura con una probabilidad del %.2f" %(no_knife*100), "% que en la imagen no se presenta un arma blanca.")
# else:
#     print("El modelo asegura con una probabilidad del %.2f" %(knife*100), "% que en la imagen se presenta un arma blanca.")
#
#
#
#

from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import messagebox
import os

root = Tk()
root.geometry("650x400+300+150")
root.title("Detector de armas blancas")
root.resizable(width=True, height=True)

def openfn():
    filename = filedialog.askopenfilename(title='Cargar imagen a identificar')
    return filename


def open_img():
    try:
        x = openfn()
        img = Image.open(x)
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(root, image=img)
        panel.image = img
        panel.pack()
        predecir = Button(root, text='Predecir', command=lambda: {prediccion(x),predecir.pack_forget(),panel.pack_forget()})
        predecir.pack()

    except:
        print("Ocurrio un error al cargar la imagen")


def prediccion(filename):
    modelo = './modelo/modelo.h5'
    pesos = './modelo/pesos.h5'

    model = load_model(modelo)
    model.load_weights(pesos)

    imagen = cv2.imread(filename)
    origen = imagen.copy()

    imagen = cv2.resize(imagen, (100, 100))
    imagen = imagen.astype('float')/255.0
    imagen = np.expand_dims(imagen, axis=0)

    (no_knife, knife) = model.predict(imagen)[0]

    if(no_knife > knife):

        mensaje = "El modelo asegura con una probabilidad del "+ str(round(no_knife*100,2)) + " que en la imagen no se presenta un arma blanca."
        messagebox.showinfo('Posible presencia de arma blanca', mensaje)
        # print("El modelo asegura con una probabilidad del %.2f" %(no_knife*100), "% que en la imagen no se presenta un arma blanca.")
    else:
        mensaje = "El modelo asegura con una probabilidad del "+ str(round(knife*100,2)) + "% que en la imagen se presenta un arma blanca."
        messagebox.showerror('Posible presencia de arma blanca', mensaje)

        # print("El modelo asegura con una probabilidad del %.2f" %(knife*100), "% que en la imagen se presenta un arma blanca.")


btn = Button(root, text='Cargar imagen', command=open_img).pack()


root.mainloop()