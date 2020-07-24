import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import cv2
from time import sleep
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
def image_capture():
	key = cv2. waitKey(1)
	webcam = cv2.VideoCapture(0)
	sleep(2)
	while True:

	    try:
	        check, frame = webcam.read()
	        print(check) #prints true as long as the webcam is running
	        print(frame) #prints matrix values of each framecd 
	        cv2.imshow("Capturing", frame)
	        key = cv2.waitKey(1)
	        if key == ord('s'): 
	            cv2.imwrite(filename='saved_img.jpg', img=frame)
	            webcam.release()
	            print("Processing image...")
	            img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
	            print("Converting RGB image to grayscale...")
	            gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
	            print("Converted RGB image to grayscale...")
	            print("Resizing image to 28x28 scale...")
	            img_ = cv2.resize(gray,(100,100))

	            print("Resized...")
	            img_resized = cv2.imwrite(filename='C:/Users/Vedant/Desktop/datas/test/saved_img.jpg', img=img_)
	            print("Image saved!")
	            
	            break
	        
	        elif key == ord('q'):
	            webcam.release()
	            cv2.destroyAllWindows()
	            break
	    
	    except(KeyboardInterrupt):
	        print("Turning off camera.")
	        webcam.release()
	        print("Camera off.")
	        print("Program ended.")
	        cv2.destroyAllWindows()
	        break
def make_prediction(captured_image):
	img=captured_image

	model = tf.keras.models.load_model('C:/Users/Vedant/Desktop/datas/liveness_mod.h5')
	model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
	img = np.reshape(img,[1,100,100,3])
	prediction=model.predict(img)
	classes=model.predict_classes(img)
	#print(prediction)
	#print(classes[0][0])
	if classes[0][0]==0:
		print("Spoofed Image")
	else:
		print("Live image")




image1=image_capture()
img = image.load_img('C:/Users/Vedant/Desktop/datas/test/saved_img.jpg',target_size=(100,100,3))
img = image.img_to_array(img)
make_prediction(img)


