import os
import config as cf
import matplotlib.pyplot as plt
import glob
import json
assert cf
import utils as ut
import cv2
from skimage import util as ul
from skimage import data, exposure, img_as_float, transform
import skimage.io as sk
import numpy as np

# Importan las imagenes de la carpeta
train = [f for f in glob.glob(os.path.join('data_mp1\\BCCD\\train','*.jpg'))]
test = [f for f in glob.glob(os.path.join('data_mp1\\BCCD\\test','*.jpg'))]
valid = [f for f in glob.glob(os.path.join('data_mp1\\BCCD\\valid','*.jpg'))]
print(len(train),len(test),len(valid))

plt.figure("Imagenes.png")
for i in range(0,8):
    #numeroAleatorio = random.randint(0,len(test)-1)
    plt.subplot(2,4,i+1)
    img = sk.imread(os.path.join(train[i]))
    imgplot = plt.imshow(img)
    plt.title("Imagen"+str(i+1))
    plt.axis('off')
plt.suptitle("Imagenes de entrenamiento")
plt.show()

# Leer el archivo JSON
var= "train"
with open(os.path.join("data_mp1","BCCD",var,"_annotations.coco.json")) as json_file:
    dataJson = json.load(json_file)

platelet, rbc, wbc = 0, 0, 0
for i in range(0,len(dataJson["annotations"])):
    var = dataJson["annotations"][i]["category_id"]
    
    if var == 1:
        platelet += var 
    elif var == 2:
        rbc += var
    elif var == 3:
        wbc += var
print("Platelet: ",platelet,"RBC: ",rbc,"WBC: ",wbc, "Total: ", platelet+rbc+wbc)


# Anotaciones  
plt.figure("Anotaciones.png")
for i in range(0,4):
    for j in range(0,4):
        if j == 0:
            img = sk.imread(os.path.join(train[i]))
        else:
            img = ut.visualize_annotations(fold='train', img_name=train[i][20:], annotations_json_name='_annotations.coco.json', interest_class= j)
        plt.subplot(4,4,(j+1)+(i*4))
        plt.imshow(img)
        plt.axis('off')
plt.suptitle("Anotaciones de entrenamiento")
plt.show()

# Separación de 16 imágenes por canal de color y ploteo de las imágenes
plt.figure("Imagen RGB.png")
j =0
for i in range(0,4):
    img = sk.imread(os.path.join(test[i]))
    b,g,r = cv2.split(img)
    plt.subplot(4,4,j+1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(4,4,j+2)
    plt.imshow(b, cmap='gray')
    plt.axis('off')
    plt.subplot(4,4,j+3)
    plt.imshow(g, cmap='gray')
    plt.axis('off')
    plt.subplot(4,4,j+4)
    plt.imshow(r, cmap='gray')
    plt.axis('off')
    j+=4
plt.suptitle("Imagenes de prueba")
plt.show()

# Transformaciones Intesidad y contraste

plt.figure("Transformaciones de Intensidad.png")
#Imagen original
img = sk.imread(os.path.join(test[6]))
image_float = img_as_float(img)
plt.subplot(2,2,1)
plt.imshow(img)
plt.title("Imagen original")
plt.axis('off')

# Transformación de inversión de intensidad
inverted_img = ul.invert(img)
plt.subplot(2,2,2)
plt.imshow(inverted_img)
plt.title("Inversión de intensidad")
plt.axis('off')

# Transformación de gamma con exposure (aumento de contraste)
gamma_corrected = exposure.adjust_gamma(img, gamma=4.3, gain= 1)
plt.subplot(2,2,3)
plt.imshow(gamma_corrected)
plt.title("Transformación Gamma")
plt.axis('off')


# Transformación logarítmica con exposure 
logarithmic_corrected = exposure.adjust_log(img, gain=1.5, inv=False)
plt.subplot(2,2,4)
plt.imshow(logarithmic_corrected)
plt.title("Transformación logarítmica")
plt.axis('off')
plt.suptitle("Transformaciones de intensidad y contraste")
plt.show()

# Transformaciones geométricas

#Imagen original
plt.figure("Transformaciones geométricas.png")
plt.subplot(2,2,1)
plt.imshow(img)
plt.title("Imagen original")
plt.axis('off')

# Transformación de rotación y translación
matrix = np.array([[np.cos(np.pi/6), -np.sin(np.pi/6), 100],
                   [np.sin(np.pi/6), np.cos(np.pi/6), -20],
                   [0, 0, 1]])
tform = transform.EuclideanTransform(matrix)
tf_img = transform.warp(img, tform.inverse)
plt.subplot(2,2,2)
plt.imshow(tf_img)
plt.title("Rotación 30° y Translación")
plt.axis('off')

# Shear
tshear = transform.AffineTransform(shear=np.pi/6)
#print("Matrix de transformación:\n", tshear.params)
shear_img = transform.warp(img, tshear.inverse)
plt.subplot(2,2,3)
plt.imshow(shear_img)
plt.title("Shear")
plt.axis('off')


# Resize
#resize_img = transform.rescale(image_float, 0.5)

imag = cv2.resize(img,(35,35),interpolation=cv2.INTER_CUBIC)
plt.subplot(2,2,4)
plt.imshow(imag)
plt.title("Resize")
plt.axis('off')
plt.suptitle("Transformaciones geométricas")
plt.show()