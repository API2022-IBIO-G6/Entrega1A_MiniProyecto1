import os
import skimage.io
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

train = [f for f in glob.glob(os.path.join('data/train','*.jpg'))]
test = [f for f in glob.glob(os.path.join('data/test','*.jpg'))]
valid = [f for f in glob.glob(os.path.join('data/valid','*.jpg'))]
img = mpimg.imread(os.path.join(test[0]))

imgplot = plt.imshow(img)

for i in range(0,8):
    #numeroAleatorio = random.randint(0,len(test)-1)
    plt.subplot(2,4,i+1)
    img = mpimg.imread(os.path.join(train[i]))
    imgplot = plt.imshow(img)
    plt.title("Imagen"+str(i+1))
plt.show()
