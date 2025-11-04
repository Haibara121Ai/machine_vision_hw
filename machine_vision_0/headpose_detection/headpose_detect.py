import os
import sys
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LANDMARK_DIR = os.path.join(BASE_DIR, 'face_landmark_detect')
if LANDMARK_DIR not in sys.path:
    sys.path.insert(0, LANDMARK_DIR)

from landmark_detect import basic_face_alignment

class PoseEstimator:
    def __init__(self, camera_matrix=None, dist_coeffs=None, image_size=(640, 480)):
        self.model_points = self._get_6_point_model()
        
        width, height = image_size
        focal_length = width * 1.2
        cx, cy = width / 2, height / 2
        
        if camera_matrix is None:
            self.camera_matrix = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float64)
        else:
            self.camera_matrix = camera_matrix
            
        if dist_coeffs is None:
            self.dist_coeffs = np.array([0.1, -0.05, 0.001, 0.001], dtype=np.float64)
        else:
            self.dist_coeffs = dist_coeffs
    
    def _get_6_point_model(self):
        model_points = [
            [0.0, 0.0, 0.0],        # 30 - 鼻尖
            [0.0, -330.0, -65.0],   # 8  - 下巴
            [-225.0, 170.0, -135.0], # 36 - 左眼
            [225.0, 170.0, -135.0],  # 45 - 右眼
            [-150.0, -150.0, -125.0], # 48 - 左嘴
            [150.0, -150.0, -125.0]   # 54 - 右嘴
        ]
        return np.array(model_points, dtype=np.float64)
    
    def get_6_point_indices(self):
        return [30, 8, 36, 45, 48, 54]
    
    def extract_6_points(self, landmarks_68):
        indices = self.get_6_point_indices()
        return np.array([landmarks_68[i] for i in indices], dtype=np.float64)
    
    def estimate_pose(self, landmarks_68):
        if landmarks_68 is None or len(landmarks_68) != 68:
            return None, None, None
        
        image_points = self.extract_6_points(landmarks_68)
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, 
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None, None, None
        
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        rotation = Rotation.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('zyx', degrees=True)
        yaw, pitch, roll = euler_angles[0], euler_angles[1], euler_angles[2]
        
        return rotation_vector, translation_vector, (yaw, pitch, roll)

def draw_pose_axes(image, rotation_vector, translation_vector, camera_matrix, dist_coeffs=None, axis_length=150):
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4,1), dtype=np.float64)
    
    axis_3d = np.float32([
        [0.0, 0.0, 0.0],
        [axis_length, 0.0, 0.0],
        [0.0, axis_length, 0.0],
        [0.0, 0.0, axis_length]
    ]).reshape(-1, 3)
    
    imgpts, _ = cv2.projectPoints(axis_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2).astype(int)
    
    origin = tuple(imgpts[0])
    x_pt = tuple(imgpts[1])
    y_pt = tuple(imgpts[2])
    z_pt = tuple(imgpts[3])
    
    cv2.arrowedLine(image, origin, x_pt, (0, 0, 255), 2)
    cv2.arrowedLine(image, origin, y_pt, (0, 255, 0), 2)
    cv2.arrowedLine(image, origin, z_pt, (255, 0, 0), 2)
    
    
    return image

def draw_6_points(image, landmarks_68, point_size=4):
    result_image = image.copy()
    indices = [30, 8, 36, 45, 48, 54]
    
    for i, idx in enumerate(indices):
        point = landmarks_68[idx]
        x, y = int(point[0]), int(point[1])
        cv2.circle(result_image, (x, y), point_size, (0, 0, 255), -1)
    
    return result_image

if __name__ == "__main__":
    image_path = 'D:/pywork/machine_vision/machine_vision_0/face_landmark_detect/for_face_detect_result.jpg'
    img = cv2.imread(image_path)
    
    if img is None:
        exit()
    
    landmarks = basic_face_alignment(image_path)
    
    if landmarks is not None and len(landmarks) > 0:
        height, width = img.shape[:2]
        estimator = PoseEstimator(image_size=(width, height))
        
        rotation_vector, translation_vector, euler_angles = estimator.estimate_pose(landmarks[0])
        
        if rotation_vector is not None:
            yaw, pitch, roll = euler_angles
            print(f"Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°, Roll: {roll:.1f}°")
            
            result_img = img.copy()
            result_img = draw_6_points(result_img, landmarks[0])
            result_img = draw_pose_axes(result_img, rotation_vector, translation_vector, estimator.camera_matrix)
            
            cv2.imshow('6-Point Pose Estimation', result_img)
            cv2.imwrite('D:/pywork/machine_vision/machine_vision_0/headpose_detection/6points_head_pose_result.jpg', result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()