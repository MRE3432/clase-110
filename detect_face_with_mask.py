# Importar la biblioteca OpenCV
import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model('keras_model.h5')
# Definir un objeto de captura de video
vid = cv2.VideoCapture(0)
  
while(True):
      
    check,frame = vid.read()

    img = cv2.resize(frame(224,224))

    test_image = np.array(img,dtype = np.float32)
    test_image = np.expand_dims(test_image,axis=0)

    normalised_image = test_image/255.0
    
    prediction = model.predict(frame)
    print ("prediccion:",prediction)
    # Mostrar el fotograma resultante
    cv2.imshow('Fotograma', frame)
      
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
vid.release()

cv2.destroyAllWindows()
