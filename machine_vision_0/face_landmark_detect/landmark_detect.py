import face_alignment
import numpy
import cv2
import sys, os

PRT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Pytorch_Retinaface'))
if PRT_DIR not in sys.path:
    sys.path.insert(0, PRT_DIR)

os.chdir(PRT_DIR)

from face_detect_test import FaceDetector

detector = FaceDetector(
    network='mobile0.25',
    trained_model='D:/pywork/machine_vision/machine_vision_0/Pytorch_Retinaface/weights/mobilenet0.25_Final.pth',
    cpu=True
)

def bound_box_generate(img_path, detector = detector):
    """
    仅使用一张照片中最大的人脸

    return: 裁剪后的人脸图像
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    dets, img_raw = detector.detect(img_path)
    
    x1f, y1f, x2f, y2f, score = dets[0,0:5]

    x1 = int(x1f)
    y1 = int(y1f)
    x2 = int(x2f)
    y2 = int(y2f)

    cropped_img = img[y1:y2, x1:x2]

    cv2.imshow('cropped_img', cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return cropped_img

if __name__ == "__main__":
    bound_box_generate("D:/pywork/machine_vision/machine_vision_0/Pytorch_Retinaface/for_face_detect.jpg")
