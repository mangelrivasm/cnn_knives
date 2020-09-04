
import cv2
import numpy as np
from keras.models import load_model

modelo = './modelo/modelo.h5'
pesos = './modelo/pesos.h5'

model = load_model(modelo)
model.load_weights(pesos)

imagen = cv2.imread('gun.jpg')
origen = imagen.copy()

imagen = cv2.resize(imagen, (100, 100))
imagen = imagen.astype('float')/255.0
imagen = np.expand_dims(imagen, axis=0)

(no_knife, knife) = model.predict(imagen)[0]

if(no_knife > knife):
    print("El modelo asegura con una probabilidad del %.2f" %(no_knife*100), "% que en la imagen no se presenta un arma blanca.")
else:
    print("El modelo asegura con una probabilidad del %.2f" %(knife*100), "% que en la imagen se presenta un arma blanca.")




