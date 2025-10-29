import cv2

img = cv2.imread('./for_test_1.png', cv2.IMREAD_COLOR)
cv2.imshow('raw_img', img)

img2 = cv2.resize(img,(600,600))
cv2.imshow('resized_img', img2)


video_1 = cv2.VideoCapture('./for_test_2.mp4')
if not video_1.isOpened():
    print("failed to open video!")
else:
    while True:
        ret, frame = video_1.read()
        if not ret:
            print("failed to read video!")
            break
        cv2.imshow('video_test', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
video_1.release()

cv2.waitKey(0)
cv2.destroyAllWindows()