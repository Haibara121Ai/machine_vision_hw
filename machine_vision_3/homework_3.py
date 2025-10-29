import cv2
import numpy as np
import time

def base_on_cv(img_l, img_r):
    min_disp = 0
    num_disp = 16 * 4
    stereo = cv2.StereoSGBM.create(
        minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 5,
        P1 = 8 * 3 * 5 ** 2,
        P2 = 32 * 3 * 5 ** 2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    disparity = stereo.compute(img_l, img_r).astype(np.float32)

    disparity_normalized = cv2.normalize(disparity,None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    return disparity_normalized

class MystereoSGBM: #重写stereoSGBM
    def __init__(self, blockSize = 5, numDisparities = 16 * 4, p1 = 8 * 3 ** 2, p2 = 32 * 3 ** 2):
        self.blockSize = blockSize
        self.numDisparities = numDisparities
        self.p1 = p1
        self.p2 = p2
    def compute(self, img_l, img_r):

        cost_volume = self._compute_cost_volume(img_l, img_r)

        aggregated_cost = self._aggregate_cost(cost_volume, img_l)

        disparity_map = np.argmin(aggregated_cost, axis=2)

        disparity_map = cv2.normalize(disparity_map.astype(np.float32), None,alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)

        return disparity_map
    def _compute_cost_volume(self, img_l, img_r): #using SAD
        h, w = img_l.shape
        cost_volume = np.full((h, w, self.numDisparities), np.inf)
        half_blocksize = self.blockSize // 2
        for i in range(half_blocksize, h - half_blocksize):
            for j in range(half_blocksize, w - half_blocksize):
                left_block = img_l[i - half_blocksize:i + half_blocksize + 1,j - half_blocksize:j + half_blocksize + 1]
                for d in range(0,self.numDisparities):
                    if j - d < half_blocksize:
                        continue
                    right_block = img_r[i - half_blocksize:i + half_blocksize + 1,
                    j - d - half_blocksize:j - d + half_blocksize + 1]
                    cost_volume[i, j, d] = np.sum(np.abs(left_block - right_block))
        return cost_volume
    def _aggregate_cost(self, cost_volume, img_l):
        h, w = img_l.shape
        aggregated_cost = np.copy(cost_volume)
        for i in range(1, h):
            for j in range(1, w):
                for d in range(self.numDisparities):
                    min_prev = np.min(aggregated_cost[i - 1, j - 1, :])
                    if abs(d - int(np.argmin(aggregated_cost[i - 1, j - 1, :]))) <= 1:
                        vary_penalty = self.p1
                    else:
                        vary_penalty = self.p2
                    aggregated_cost[i-1, j-1, d] += min_prev + vary_penalty
        return aggregated_cost


if __name__ == '__main__':
    img_left = cv2.imread("D:/pywork/machine_vision/machine_vision_3/tsukuba_l.png", cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread("D:/pywork/machine_vision/machine_vision_3/tsukuba_r.png", cv2.IMREAD_GRAYSCALE)
    time_start = time.time()

    img_result = base_on_cv(img_left, img_right)
    # my_stereo = MystereoSGBM()
    # img_result = my_stereo.compute(img_left, img_right)

    time_end = time.time()

    cv2.imshow('Left Image', img_left)
    cv2.imshow('Right Image', img_right)
    cv2.imshow('Processed Image', img_result)

    print(f"time cost: {time_end - time_start}")
    cv2.waitKey(0)
