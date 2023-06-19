import requests
import ftplib
from pathlib import Path
from datetime import datetime
import subprocess
import RPi.GPIO as GPIO
from time import sleep
import os
import time
from board import SCL, SDA
import busio
from oled_text import OledText, Layout64, BigLine, SmallLine
i2c = busio.I2C(SCL, SDA)

# Create the display, pass its pixel dimensions
oled = OledText(i2c, 128, 64)

oled.layout = {
    1: BigLine(0, 0, font="Arimo.ttf", size=17),
    2: BigLine(22, 14, font="Arimo.ttf", size=17),
    3: BigLine(0, 18, font="FreeSans.ttf", size=20),
}



from_cloud = 'http://myphptestfiles.000webhostapp.com/skinRead.php'
headers = {'Content-Type': 'application/x-www-form-urlencoded'}



GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

buttonPin = 26
GPIO.setup(buttonPin, GPIO.IN,pull_up_down=GPIO.PUD_DOWN)



led =21
GPIO.setup(led, GPIO.OUT)
GPIO.output(led, GPIO.LOW)


ftpUser = 'myphptestfiles'
ftpServer = 'files.000webhost.com'
ftpPassword = 'server@12345'


imagePathOnRaspberry ='/home/pi/ftp/image.jpeg'
imagePathOnServer = 'skin'
filename = Path('image.jpeg')
         
         
def check_msg():
    global txt
        # Making a GET request
    r = requests.get(from_cloud,headers=headers)
    # check status code for response received
    # success code - 200
    txt = r.text
    txt = txt[28:45]
    print(txt)
    
            
def captureImage():
    print("capture image")
    GPIO.output(led, GPIO.HIGH)
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
    GPIO.output(led, GPIO.LOW)
    
    
while True:
    buttonState = GPIO.input(buttonPin)
    if (buttonState == True):
        captureImage()
        
    check_msg()
    
    oled.text("Skin Cancer",1)
    oled.text("      Detection",2)
    oled.text(txt, 3)
          
time.sleep(2)