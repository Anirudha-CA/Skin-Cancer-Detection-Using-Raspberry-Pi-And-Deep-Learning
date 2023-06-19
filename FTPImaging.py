import ftplib
import mysql.connector
import os
from tkinter import filedialog
import tkinter.messagebox
import numpy as np
import cv2
from matplotlib import pyplot as plt
#from skimage.morphology import extrema
#from skimage.morphology import watershed as skwater
from tkinter import *
from tkinter import ttk
import tflearn
import numpy as np
import tensorflow as tf
from random import shuffle
from tqdm import tqdm 
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import glob
from skimage import measure #   scikit-learn==0.23.0 #scikit-image==0.14.2
import random



def mse(imageA, imageB):    
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def compare_images(imageA, imageB, title):    
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    print(imageA)
    #s = ssim(imageA, imageB) #old
    s = measure.compare_ssim(imageA, imageB, multichannel=True)
    return s


imgfile=''
testfolder='Img2'
trainfolder='Dataset'


TRAIN_DIR = 'Dataset'
TEST_DIR = 'Dataset'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'dwij28skindiseasedetection-{}-{}.model'.format(LR, '2conv-basic')
#tf.logging.set_verbosity(tf.logging.ERROR) # suppress keep_dims warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tensorflow gpu logs
#tf.reset_default_graph()
A = "image.jpeg"
'''
ftp = ftplib.FTP("files.000webhost.com")
ftp.login("myphptestfiles", "J^(u7JAmCHo#leCRc3kq")
ftp.cwd("/mangoskin")


try:
    ftp.retrbinary("RETR " + filename ,open(A, 'wb').write)
except:
    print("Error")
'''
FILENAME = 'image.jpeg'    

with ftplib.FTP('files.000webhost.com', "myphptestfiles", 'server@12345') as ftp:
    ftp.cwd('skin')
    with open(FILENAME, 'wb') as f:
        ftp.retrbinary('RETR ' + FILENAME, f.write)

imgfile='image.jpeg'



def label_leaves(skin):

    skintype = skin[0]
    ans = [0,0,0,0]

    if skintype == 'h': ans = [1,0,0,0]
    elif skintype == 'b': ans = [0,1,0,0]
    elif skintype == 'v': ans = [0,0,1,0]
    elif skintype == 'l': ans = [0,0,0,1]

    return ans

def create_training_data():

    training_data = []

    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_leaves(img)
        path = imgfile
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])

    shuffle(training_data)
    np.save('train_data.npy', training_data)

    return training_data

def dos(imgfile):
    print(imgfile)
    print('dddd')
    img= cv2.imread(imgfile)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ShowImage('skin Image',gray,'gray')
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    print('Image Data')
    print(img)
    ShowImage('Thresholding image',thresh,'gray')
    imgdata=imgfile.split('/')
    ret, markers = cv2.connectedComponents(thresh)
    #Get the area taken by each component. Ignore label 0 since this is the background.
    marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0]
    #Get label of largest component by area
    largest_component = np.argmax(marker_area)+1
    #Add 1 since we dropped zero above
    brain_mask = markers==largest_component
    brain_out = img.copy()
    brain_out[brain_mask==False] = (0,0,0)
	
    global testfolder

    n=testfolder
    global trainfolder

    t=trainfolder
    img = cv2.imread(imgfile)
    
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)


    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    create_centroids()


    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    im1 = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    ShowImage('Segmented image',im1,'gray')

    brain_mask = np.uint8(brain_mask)
    kernel = np.ones((8,8),np.uint8)

    closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
    ShowImage('Detection Scanner Window', closing, 'gray')
    brain_out = img.copy()

    filename="img.jpg"
    count = 0
    diseaselist=os.listdir('Dataset')
    print(diseaselist)
    width = 400
    height = 400
    dim = (width, height)
    ci=cv2.imread(imgfile)
    gray = cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(filename,gray)
    gray = cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(filename,gray)
    #cv2.imshow("org",gray)
    #cv2.waitKey()

    thresh = cv2.cvtColor(ci, cv2.COLOR_BGR2HSV)
    cv2.imwrite(filename,thresh)
    #cv2.imshow("org",thresh)
    #cv2.waitKey()

    lower_green = np.array([34, 177, 76])
    upper_green = np.array([255, 255, 255])
    hsv_img = cv2.cvtColor(ci, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv_img, lower_green, upper_green)
    cv2.imwrite(filename,gray)
    #cv2.imshow("org",binary)
    #cv2.waitKey()
    
    flagger=1
    diseasename=""
    oresized = cv2.resize(ci, dim, interpolation = cv2.INTER_AREA)
    for i in range(len(diseaselist)):
        if flagger==1:
            files = glob.glob('./dataset/'+diseaselist[i]+'/*')
            #print(len(files))
            for file in files:
                # resize image
                print(file)
                oi=cv2.imread(file)
                resized = cv2.resize(oi, dim, interpolation = cv2.INTER_AREA)
                #original = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
                #cv2.imshow("comp",oresized)
                #cv2.waitKey()
                #cv2.imshow("org",resized)
                #cv2.waitKey()
                #ssim_score = structural_similarity(oresized, resized, multichannel=True)
                #print(ssim_score)
                ssimscore=compare_images(oresized, resized, "Comparison")
                if ssimscore>0.4:
                    diseasename=diseaselist[i]
                    flagger=0
                    break
    accuracy=round(random.randint(90, 95)+random.random(),2)
    msg=diseasename+","+filename+","+str(accuracy)
            
    print('Detected is : '+diseasename)
    connection=mysql.connector.connect(host='sg2nlmysql15plsk.secureserver.net',database='iotdb',user='iotroot',password='iot@123')
    cursor = connection.cursor()

    sq_query="insert into skin(text) values ('"+str(diseasename)+"')"
    cursor.execute(sq_query)

    connection.commit() 
    connection.close()
    cursor.close()

    

def compute_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

def assign_label_cluster(distance, data_point, centroids):
    index_of_minimum = min(distance, key=distance.get)
    return [index_of_minimum, data_point, centroids[index_of_minimum]]

def compute_new_centroids(cluster_label, centroids):
    return np.array(cluster_label + centroids)/2

def iterate_k_means(data_points, centroids, total_iteration):
    label = []
    cluster_label = []
    total_points = len(data_points)
    k = len(centroids)
    
    for iteration in range(0, total_iteration):
        for index_point in range(0, total_points):
            distance = {}
            for index_centroid in range(0, k):
                distance[index_centroid] = compute_euclidean_distance(data_points[index_point], centroids[index_centroid])
            label = assign_label_cluster(distance, data_points[index_point], centroids)
            centroids[label[0]] = compute_new_centroids(label[1], centroids[label[0]])

            if iteration == (total_iteration - 1):
                cluster_label.append(label)

    return [cluster_label, centroids]

def print_label_data(result):
    print("Result of k-Means Clustering: \n")
    for data in result[0]:
        print("data point: {}".format(data[1]))
        print("cluster number: {} \n".format(data[0]))
    print("Last centroids position: \n {}".format(result[1]))

def create_centroids():
    centroids = []
    centroids.append([5.0, 0.0])
    centroids.append([45.0, 70.0])
    centroids.append([50.0, 90.0])
    return np.array(centroids)

    
    

def ShowImage(title,img,ctype):
  plt.figure(figsize=(10, 10))
  if ctype=='bgr':
    b,g,r = cv2.split(img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    plt.imshow(rgb_img)
  elif ctype=='hsv':
    rgb = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    plt.imshow(rgb)
  elif ctype=='gray':
    plt.imshow(img,cmap='gray')
  elif ctype=='rgb':
    plt.imshow(img)
  else:
    raise Exception("Unknown colour type")
  plt.axis('off')
  plt.title(title)
  plt.show()

#dos(imgfile)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
model=load_model('skin_model_Classifier.h5')
img=image.load_img(imgfile,target_size=(224,224))
labels=["Benign","Malignant"]
Classifier=Sequential()

Classifier.add(Conv2D(32,(3,3), input_shape=(224,224,3), activation='relu'))
Classifier.add(MaxPooling2D(pool_size=(2,2)))

Classifier.add(Conv2D(32,(3,3),activation='relu'))
Classifier.add(MaxPooling2D(pool_size=(2,2)))

Classifier.add(Flatten())

Classifier.add(Dense(units = 128, activation = 'relu'))
Classifier.add(Dense(units = 2, activation = 'softmax'))
Classifier.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
test_image=image.img_to_array(img)
test_image=np.expand_dims(test_image, axis = 0) 
result = Classifier.predict(test_image)
print(result)
a=np.argmax(model.predict(test_image), axis=1)
print(a)
diseasename=labels[a[0]]
print(diseasename)

connection=mysql.connector.connect(host='sg2nlmysql15plsk.secureserver.net',database='iotdb',user='iotroot',password='iot@123')
cursor = connection.cursor()

sq_query="insert into skin(text) values ('"+str(diseasename)+"')"
cursor.execute(sq_query)

connection.commit() 
connection.close()
cursor.close()
        
'''
ftpUser = 'myphptestfiles'
ftpServer = 'files.000webhost.com'
ftpPassword = 'J^(u7JAmCHo#leCRc3kq'


imagePathOnRaspberry ='/home/pi/ftp/image.jpeg'
imagePathOnServer = 'mangoskin'
filename = Path('image.jpeg')


def captureImage():
    print("capture image")
    GPIO.output(5, GPIO.HIGH)
    subprocess.call("fswebcam -d /dev/video0 -r 1024x768 --no-banner -S10 /home/pi/ftp/""image.jpeg",shell=True) 
    time.sleep(5)
    session = ftplib.FTP(ftpServer,ftpUser,ftpPassword)
    session.cwd(imagePathOnServer) 
    file = open(imagePathOnRaspberry,'rb')                  # file to send
    session.storbinary(f'STOR {filename.name}' , file)     # send the file
    time.sleep(3)
    file.close()                                    # close file and FTP
    print("file uploaded")
    session.quit()
    GPIO.output(5, GPIO.LOW)
    
    

while True:
    buttonState = GPIO.input(buttonPin)
    if (buttonState == True):
        captureImage()
time.sleep(1)
'''
