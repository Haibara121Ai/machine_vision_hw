import cv2
import numpy as np
import matplotlib.pyplot as plt

class HeadPoseEstimator5Points:
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        """
        5点头部姿态估计器
        点顺序: ['left eye', 'right eye', 'nose', 'left mouth corner', 'right mouth corner']
        """
        self.model_points_3d = np.array([
            [-80, -50, -50],
            [80, -50, -50],
            [0.0, 0.0, 0.0],
            [-80, 120, -20],
            [80, 120, -20]
        ], dtype=np.float64)

        if camera_matrix is None:
            self.camera_matrix = np.array([
                [1000, 0, 320],
                [0, 1000, 240],
                [0, 0, 1]
            ], dtype=np.float64)
        else:
            self.camera_matrix = camera_matrix

        if dist_coeffs is None:
            self.dist_coeffs = np.zeros((4, 1))
        else:
            self.dist_coeffs = dist_coeffs

        self.point_names = ['left eye', 'right eye', 'nose', 'left mouth corner', 'right mouth corner']

    def estimate_pose(self, points_2d):
        """
        使用5个2D点估计头部姿态

        Args:
            points_2d: 5个2D点坐标，顺序为 ['left eye', 'right eye', 'nose', 'left mouth corner', 'right mouth corner']

        Returns:
            dict: 包含姿态信息的字典
        """
        points_2d = np.array(points_2d, dtype=np.float64)

        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points_3d,
                points_2d,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP
            )

            if not success:
                return {'success': False, 'error': 'solvePnP failed'}

            euler_angles = self.rotation_vector_to_euler(rotation_vector)

            projected_points, _ = cv2.projectPoints(
                self.model_points_3d, rotation_vector, translation_vector,
                self.camera_matrix, self.dist_coeffs
            )
            reprojection_error = np.mean(np.linalg.norm(points_2d - projected_points.reshape(-1, 2), axis=1))

            return {
                'success': True,
                'rotation_vector': rotation_vector,
                'translation_vector': translation_vector,
                'euler_angles': euler_angles,  # [yaw, pitch, roll]
                'yaw': euler_angles[0],
                'pitch': euler_angles[1],
                'roll': euler_angles[2],
                'reprojection_error': reprojection_error,
                'projected_points': projected_points.reshape(-1, 2)
            }

        except Exception as e:
            print(f"Error: {e}")
            return {'success': False, 'error': str(e)}

    def rotation_vector_to_euler(self, rotation_vector):
        """将旋转向量转换为欧拉角度"""
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] +
                     rotation_matrix[1, 0] * rotation_matrix[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # yaw
            y = np.arctan2(-rotation_matrix[2, 0], sy)  # pitch
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # roll
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0

        yaw = np.degrees(x)  # 左右摇头
        pitch = np.degrees(y)  # 上下点头
        roll = np.degrees(z)  # 头部倾斜

        return [yaw, pitch, roll]

    def draw_pose_on_image(self, image, points_2d, pose_result, draw_axes=True, draw_points=True):
        """
        在原图上绘制头部姿态结果

        Args:
            image: 原始图像 (BGR格式)
            points_2d: 5个特征点坐标
            pose_result: 姿态估计结果
            draw_axes: 是否绘制坐标轴
            draw_points: 是否绘制特征点

        Returns:
            numpy.ndarray: 绘制了姿态结果的图像
        """
        result_image = image.copy()

        if pose_result['success']:
            if draw_points:
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
                for i, point in enumerate(points_2d):
                    color = colors[i]
                    cv2.circle(result_image, tuple(point.astype(int)), 8, color, -1)
                    cv2.circle(result_image, tuple(point.astype(int)), 8, (255, 255, 255), 2)
                    cv2.putText(result_image, self.point_names[i],
                                (int(point[0]) + 12, int(point[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if draw_axes:
                nose_point = points_2d[2]

                img_height, img_width = image.shape[:2]
                axis_length = min(img_width, img_height) * 0.15  # 轴长为图像较小边长的15%

                axis_points_3d = np.float32([
                    [0, 0, 0],
                    [axis_length, 0, 0],  # X轴
                    [0, axis_length, 0],  # Y轴
                    [0, 0, -axis_length]  # Z轴
                ])

                axis_points_2d, _ = cv2.projectPoints(
                    axis_points_3d,
                    pose_result['rotation_vector'],
                    pose_result['translation_vector'],
                    self.camera_matrix,
                    self.dist_coeffs
                )

                origin = tuple(nose_point.astype(int))

                axis_thickness = max(4, int(min(img_width, img_height) * 0.005))

                result_image = cv2.line(result_image, origin,
                                        tuple(axis_points_2d[1].ravel().astype(int)),
                                        (0, 0, 255), axis_thickness)  # X-红
                result_image = cv2.line(result_image, origin,
                                        tuple(axis_points_2d[2].ravel().astype(int)),
                                        (0, 255, 0), axis_thickness)  # Y-绿
                result_image = cv2.line(result_image, origin,
                                        tuple(axis_points_2d[3].ravel().astype(int)),
                                        (255, 0, 0), axis_thickness)  # Z-蓝

                # 添加坐标轴标签
                font_scale = max(0.5, min(img_width, img_height) * 0.001)
                font_thickness = max(1, int(font_scale * 2))

                cv2.putText(result_image, 'X',
                            tuple(axis_points_2d[1].ravel().astype(int) + np.array([10, 10])),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
                cv2.putText(result_image, 'Y',
                            tuple(axis_points_2d[2].ravel().astype(int) + np.array([10, 10])),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
                cv2.putText(result_image, 'Z',
                            tuple(axis_points_2d[3].ravel().astype(int) + np.array([10, 10])),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness)

            text_lines = [
                f"Yaw: {pose_result['yaw']:.2f}degrees",
                f"Pitch: {pose_result['pitch']:.2f}degrees",
                f"Roll: {pose_result['roll']:.2f}degrees",
                f"Reprojection Error: {pose_result['reprojection_error']:.2f}px"
            ]

            text_bg_height = len(text_lines) * 35 + 20
            text_bg_width = 350

            overlay = result_image.copy()
            cv2.rectangle(overlay, (10, 10), (text_bg_width, text_bg_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, result_image, 0.4, 0, result_image)

            cv2.rectangle(result_image, (10, 10), (text_bg_width, text_bg_height), (255, 255, 255), 2)

            font_scale = 0.7
            font_thickness = 2
            for i, text in enumerate(text_lines):
                y_position = 40 + i * 35
                cv2.putText(result_image, text, (20, y_position),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thickness)

        return result_image


def estimate_pose_on_real_image(image_path, points_2d, output_path=None, camera_matrix=None):
    """
    在真实图像上估计头部姿态并可视化

    Args:
        image_path: 图像文件路径
        points_2d: 5个特征点坐标
        output_path: 输出图像路径
        camera_matrix: 相机矩阵

    Returns:
        dict: 姿态估计结果
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image: {image_path}")
        return None

    height, width = image.shape[:2]
    print(f"Image size: {width}x{height}")

    if camera_matrix is None:
        focal_length = max(width, height)
        camera_matrix = np.array([
            [focal_length, 0, width // 2],
            [0, focal_length, height // 2],
            [0, 0, 1]
        ], dtype=np.float64)
        print(f"Auto-adjusted camera matrix:")
        print(camera_matrix)

    pose_estimator = HeadPoseEstimator5Points(camera_matrix=camera_matrix)

    points_2d = np.array(points_2d, dtype=np.float64)
    for i, point in enumerate(points_2d):
        if point[0] < 0 or point[0] > width or point[1] < 0 or point[1] > height:
            print(f"Warning: Point {i} ({point}) is outside image boundaries")

    pose_result = pose_estimator.estimate_pose(points_2d)

    if pose_result['success']:
        print("\n=== Head Pose Estimation Results ===")
        print(f"Yaw (left-right): {pose_result['yaw']:.2f}°")
        print(f"Pitch (up-down): {pose_result['pitch']:.2f}°")
        print(f"Roll (head tilt): {pose_result['roll']:.2f}°")
        print(f"Reprojection Error: {pose_result['reprojection_error']:.2f} px")

        result_image = pose_estimator.draw_pose_on_image(image, points_2d, pose_result)

        plt.figure(figsize=(16, 8))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title("Head Pose Estimation Result")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"Result image saved: {output_path}")

        return pose_result
    else:
        print("Pose estimation failed!")
        if 'error' in pose_result:
            print(f"Error: {pose_result['error']}")
        return None


def demo_with_real_image():
    """使用真实图像演示头部姿态估计"""

    image_path = "for_face_detect.jpg"

    your_points_2d = np.array([
        [305.64, 468.30],  # left eye
        [441.09, 492.03],  # right eye
        [379.02, 576.13],  # nose (corrected from 2576.13)
        [287.90, 609.81],  # left mouth corner
        [404.17, 630.44]  # right mouth corner
    ], dtype=np.float64)

    print("Input points:")
    point_names = ['left eye', 'right eye', 'nose', 'left mouth corner', 'right mouth corner']
    for i, point in enumerate(your_points_2d):
        print(f"  {point_names[i]}: ({point[0]:.1f}, {point[1]:.1f})")

    pose_result = estimate_pose_on_real_image(
        image_path=image_path,
        points_2d=your_points_2d,
        output_path="head_pose_result.jpg"
    )

    return pose_result


def batch_process_images(image_points_dict, output_dir="results"):
    """
    批量处理多张图像

    Args:
        image_points_dict: 字典，键为图像路径，值为特征点坐标
        output_dir: 输出目录
    """
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = {}

    for image_path, points_2d in image_points_dict.items():
        print(f"\n{'=' * 50}")
        print(f"Processing: {image_path}")
        print(f"{'=' * 50}")

        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"pose_{filename}")

        # 处理单张图像
        result = estimate_pose_on_real_image(
            image_path=image_path,
            points_2d=points_2d,
            output_path=output_path
        )

        results[image_path] = result

    return results


def test_different_poses():
    """测试不同的头部姿态"""

    frontal_pose = np.array([
        [300, 250],  # left eye
        [400, 250],  # right eye
        [350, 300],  # nose
        [320, 350],  # left mouth
        [380, 350]  # right mouth
    ], dtype=np.float64)

    left_pose = np.array([
        [320, 250],  # left eye
        [420, 250],  # right eye
        [370, 300],  # nose
        [340, 350],  # left mouth
        [400, 350]  # right mouth
    ], dtype=np.float64)

    test_cases = {
        "frontal": frontal_pose,
        "left_turn": left_pose
    }

    for case_name, points in test_cases.items():
        print(f"\nTesting {case_name} pose...")

        test_image = np.ones((600, 800, 3), dtype=np.uint8) * 100

        pose_estimator = HeadPoseEstimator5Points()
        result = pose_estimator.estimate_pose(points)

        if result['success']:
            result_image = pose_estimator.draw_pose_on_image(test_image, points, result)

            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            plt.title(
                f"Head Pose: {case_name}\nYaw: {result['yaw']:.1f}°, Pitch: {result['pitch']:.1f}°, Roll: {result['roll']:.1f}°")
            plt.axis('off')
            plt.show()


if __name__ == "__main__":
    result = demo_with_real_image()
