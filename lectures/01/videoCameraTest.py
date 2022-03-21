import cv2

cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('./samplevideo.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0,
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while True:
    ret, frame = cap.read()

    if ret == False:
        break

    cv2.rectangle(frame, (frame.shape[1] // 2 - 50, frame.shape[0] // 2 - 50),
                  (frame.shape[1] // 2 + 50, frame.shape[0] // 2 + 50), (0, 0, 255))

    out.write(frame)
    cv2.imshow('capture', frame)

    key = cv2.waitKey(20) & 0xFF

    if key == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()