import face_alignment
import numpy
import cv2
import sys, os
from pathlib import Path

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
    stem = Path(img_path).stem

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    dets, img_raw = detector.detect(img_path)
    
    x1f, y1f, x2f, y2f, score = dets[0,0:5]
    # 对边界框进行适当扩展
    if x1f > 40:
        x1f -= 40
    if y1f > 40:
        y1f -= 40
    if x2f < img.shape[1] - 40:
        x2f += 40
    if y2f < img.shape[0] - 40:
        y2f += 40

    x1 = int(x1f)
    y1 = int(y1f)
    x2 = int(x2f)
    y2 = int(y2f)

    cropped_img = img[y1:y2, x1:x2]

    cv2.imshow('cropped_img', cropped_img)
    cv2.imwrite(f'D:/pywork/machine_vision/machine_vision_0/face_landmark_detect/{stem}_result.jpg', cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return cropped_img

def basic_face_alignment(img_path):
    # 初始化face-alignment检测器
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu')

    stem = Path(img_path).stem
    
    img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    landmarks = fa.get_landmarks(rgb_img)
    
    if landmarks is not None:
        print(f"检测到 {len(landmarks)} 张人脸")
        
        for i, face_landmarks in enumerate(landmarks):
            print(f"人脸 {i+1}: {len(face_landmarks)} 个关键点")
            
            for point in face_landmarks:
                x, y = int(point[0]), int(point[1])
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        
        cv2.imshow('Face Alignment Results', img)
        cv2.imwrite(f'D:/pywork/machine_vision/machine_vision_0/face_landmark_detect/{stem}_aligned_result.jpg', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return landmarks
    else:
        print("未检测到人脸")

if __name__ == "__main__":
    bound_box_generate('D:/pywork/machine_vision/machine_vision_0/Pytorch_Retinaface/for_face_detect.jpg')
    basic_face_alignment('D:/pywork/machine_vision/machine_vision_0/face_landmark_detect/for_face_detect_result.jpg')
