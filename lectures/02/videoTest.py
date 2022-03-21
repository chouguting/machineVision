import cv2
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret == False:
        break
    cv2.imshow('capture', frame)

    key = cv2.waitKey(20) & 0xFF

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print('finish')