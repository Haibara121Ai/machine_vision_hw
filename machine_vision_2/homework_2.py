import cv2
import numpy as np
import glob
import os

cube_len = 30
chessboard_size  = (8, 6)

objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp = objp * cube_len

# print(f"objp:{objp}")

objpoints = []
imgpoints = []

image_files = glob.glob('D:/pywork/machine_vision/machine_vision_2/chessboard_img/*.ppm')
for image_file in image_files:
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

# img = cv2.imread(f"D:/pywork/machine_vision/machine_vision_2/chessboard_img/c-0000.ppm", cv2.IMREAD_GRAYSCALE)
# cv2.imshow('raw_img',img)
    ret, corners = cv2.findChessboardCorners(img, chessboard_size, None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        # print(f"corners2：{corners2}")
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)

        objpoints.append(objp)
        imgpoints.append(corners2)

        for i, corner in enumerate(corners2):
            corner_int = np.array(corner)
            corner_int = tuple(map(int, corner_int.ravel()))
            # cv2.putText(img, str(i + 1), corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.imshow('Chessboard Corners', img)
        # while True:
        #     if cv2.waitKey(100) == 27:
        #         break
    # cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)

print("ret:",ret)
print("intrinsic matrix:\n",mtx)
print("distortion:\n",dist)   # 畸变系数 (k_1,k_2,k_3,p1,p2)
print("Rotation vector:\n",rvecs)   # 旋转向量 (R)
print("Translation vector:\n",tvecs)  # 平移向量 (T)

output_dir = "./img_saved/"
os.makedirs(output_dir, exist_ok=True)

for i, image_file in enumerate(image_files):
    img = cv2.imread(image_file)
    if img is None:
        continue

    undistorted_img = cv2.undistort(img, mtx, dist)

    filename = os.path.basename(image_file)
    output_path = os.path.join(output_dir, f"undistorted_{filename}")
    cv2.imwrite(output_path, undistorted_img)

print("finished！")
cv2.destroyAllWindows()