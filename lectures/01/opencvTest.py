import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

url = 'https://tronclass.ntou.edu.tw:443/api/uploads/3206911/in-rich-content?created_at=2022-02-13T07:24:53Z'
src_file = tf.keras.utils.get_file('highway', origin=url)
img = cv2.imread(src_file)
# if cv2.imread cannot find the image file, it returns None
if img is not None:
    # show this image by cv2.imshow
    cv2.imshow('image', img)

    # call cv2.waitKey to process window messages
    cv2.waitKey()

    # destroy all windows
    cv2.destroyAllWindows()
else:
    print('image file is not found')