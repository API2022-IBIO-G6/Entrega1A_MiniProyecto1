import os
import skimage.io
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import json
import utils as ut

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
    print(train[0])
    imgplot = plt.imshow(img)
    plt.title("Imagen"+str(i+1))
    plt.axis('off')
plt.suptitle("Imagenes de entrenamiento")
plt.show()

# Leer el archivo JSON
with open(os.path.join("data\\train","_annotations.coco.json")) as json_file:
    data = json.load(json_file)
    
print(ut.visualize_annotations(fold='train', img_name=train[0], annotations_json_name='_annotations.coco.json', interest_class=-1))