import numpy as np
import cv2

good_angle = {
    'yaw': 60,
    'pitch': 20,
    'roll': 20
}

def simple_face_align_complete(image, landmarks):
    """
    完整的人脸对齐函数（包含眼睛、鼻子、嘴巴的对齐）
    """
    if image is None or landmarks is None or len(landmarks) != 68:
        return None, None
    
    landmarks = np.array(landmarks, dtype=np.float32)
    
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)
    nose_tip = landmarks[30]  # 鼻尖
    mouth_center = np.mean(landmarks[48:68], axis=0)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    face_center = np.mean([left_eye, right_eye, nose_tip, mouth_center], axis=0)
    
    eye_dist = np.sqrt(dx*dx + dy*dy)
    target_eye_dist = 80
    scale = target_eye_dist / eye_dist  * 1.8
    
    M = cv2.getRotationMatrix2D(tuple(face_center), angle, scale)

    height, width = image.shape[:2]
    aligned_image = cv2.warpAffine(image, M, (width, height))
    
    ones = np.ones((len(landmarks), 1))
    landmarks_homo = np.hstack([landmarks, ones])
    aligned_landmarks = (M @ landmarks_homo.T).T
    
    return aligned_image, aligned_landmarks

def visualize_alignment(original_img, original_landmarks, aligned_img, aligned_landmarks):
    """可视化对齐结果
    
    args:
        original_img:原图像
        original_landmarks:原特征点
        aligned_img:对齐后的图像
        aligned_landmarks:对齐后的特征点
    """

    height = max(original_img.shape[0], aligned_img.shape[0])
    width = original_img.shape[1] + aligned_img.shape[1]
    
    comparison = np.zeros((height, width, 3), dtype=np.uint8)
    
    h1, w1 = original_img.shape[:2]
    comparison[0:h1, 0:w1] = original_img
    
    h2, w2 = aligned_img.shape[:2]
    comparison[0:h2, w1:w1+w2] = aligned_img
    
    for point in original_landmarks:
        x, y = int(point[0]), int(point[1])
        cv2.circle(comparison, (x, y), 2, (0, 255, 0), -1)
    
    for point in aligned_landmarks:
        x, y = int(point[0] + w1), int(point[1])
        cv2.circle(comparison, (x, y), 2, (0, 0, 255), -1)
    
    cv2.putText(comparison, 'Original', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.putText(comparison, 'Aligned', (w1 + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    
    return comparison

class pose_for_appliance:
    def __init__(self, angle):
        self.yaw = angle[0]
        self.pitch = angle[1]
        self.roll = angle[2]
    def is_good_pose(self):
        if (abs(self.yaw) <= good_angle['yaw'] and
            abs(self.pitch) <= good_angle['pitch'] and
            abs(self.roll) <= good_angle['roll']):
            return True
    def is_front_pose(self):
        if (abs(self.yaw) <= 30 and
            abs(self.pitch) <= 20 and
            abs(self.roll) <= 20):
            return True




def test_face_alignment():
    """测试人脸对齐"""
    image = cv2.imread('D:/pywork/machine_vision/machine_vision_0/face_landmark_detect/for_face_detect_result.jpg')
    
    landmarks = [
        [82,158],[76,197],[71,236],[71,275],[76,320],[99,364],[121,403],[155,442],
        [199,465],[244,459],[266,437],[283,414],[311,387],[333,348],[350,320],
        [367,286],[372,247],[155,169],[182,169],[205,175],[227,180],[244,191],
        [322,208],[339,202],[355,197],[372,202],[378,208],[272,247],[272,281],
        [272,308],[266,336],[227,336],[238,342],[249,348],[266,342],[272,342],
        [171,214],[194,208],[210,214],[227,230],[210,230],[188,225],[300,236],
        [322,230],[344,230],[350,242],[339,247],[316,247],[182,370],[205,370],
        [233,370],[244,375],[255,375],[272,381],[277,381],[266,398],[249,409],
        [233,409],[216,403],[199,392],[188,370],[221,381],[238,387],[249,387],
        [272,381],[249,387],[238,387],[221,387]
    ]

    aligned_img, aligned_landmarks = simple_face_align_complete(image, landmarks)
    
    if aligned_img is not None:
        result = visualize_alignment(image, landmarks, aligned_img, aligned_landmarks)
        cv2.imshow('Face Alignment Result', result)
        cv2.imwrite('D:/pywork/machine_vision/machine_vision_0/estimation_for_appliance/face_alignment_result.jpg', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("人脸对齐完成！")
        print(f"原始图像尺寸: {image.shape}")
        print(f"对齐后尺寸: {aligned_img.shape}")

if __name__ == "__main__":
    test_face_alignment()