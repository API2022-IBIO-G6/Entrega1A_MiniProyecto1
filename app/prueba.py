import os
import config as cf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import json
assert cf
import utils as ut
import cv2
import random




# Importan las imagenes de la carpeta
train = [f for f in glob.glob(os.path.join('data\\train','*.jpg'))]
test = [f for f in glob.glob(os.path.join('data\\test','*.jpg'))]
valid = [f for f in glob.glob(os.path.join('data\\valid','*.jpg'))]

# Generar el primer subplot
#img = mpimg.imread(os.path.join(test[0]))

#imgplot = plt.imshow(img)

for i in range(0,8):
    #numeroAleatorio = random.randint(0,len(test)-1)
    plt.subplot(2,4,i+1)
    img = mpimg.imread(os.path.join(train[i]))
    imgplot = plt.imshow(img)
    plt.title("Imagen"+str(i+1))
    plt.axis('off')
plt.suptitle("Imagenes de entrenamiento")
plt.show()

# Leer el archivo JSON
var= "train"
with open(os.path.join("data",var,"_annotations.coco.json")) as json_file:
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


    
ima6 = ut.visualize_annotations(fold='train', img_name=train[100], annotations_json_name='_annotations.coco.json', interest_class=-1)
lol = plt.imshow(ima6)
plt.show()

# Separación de 16 imágenes por canal de color y ploteo de las imágenes

"""
realizar un subplot de 4x4 donde la primera columna corresponde
a la imagen original y las otras 3 columnas corresponden a los 
diferentes canales que componen la imagen.
"""

j =0
for i in range(0,4):
    img = cv2.imread(os.path.join(test[i]))
    print(ut.pred_score(img))
    b,g,r = cv2.split(img)
    plt.subplot(4,4,j+1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(4,4,j+2)
    plt.imshow(b)
    #plt.title("Blue")
    plt.axis('off')
    plt.subplot(4,4,j+3)
    plt.imshow(g)
    #plt.title("Green")
    plt.axis('off')
    plt.subplot(4,4,j+4)
    plt.imshow(r)
    #plt.title("Red")
    plt.axis('off')
    j+=4
    
plt.suptitle("Imagenes de prueba")
plt.show()


