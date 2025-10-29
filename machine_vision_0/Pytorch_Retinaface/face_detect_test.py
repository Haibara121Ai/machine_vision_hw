import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.append('./machine_vision_0/Pytorch_Retinaface')
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.config import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms



class FaceDetector:
    def __init__(self, network='resnet50',
                 trained_model='D:/pywork/machine_vision/machine_vision_0/Pytorch_Retinaface/weights/Resnet50_Final.pth',
                 cpu=False):
        """
        初始化人脸检测器
        Args:
            network: 网络类型 'mobilenet' 或 'resnet50'
            trained_model: 预训练权重路径
            cpu: 是否使用CPU
        """
        torch.set_grad_enabled(False)

        # 选择网络配置
        if network == "mobile0.25":
            self.cfg = cfg_mnet
        elif network == "resnet50":
            self.cfg = cfg_re50

        # 设置设备
        self.device = torch.device("cpu" if cpu else "cuda")

        # 初始化模型
        self.net = RetinaFace(cfg=self.cfg, phase='test')
        self.net = self.load_model(self.net, trained_model, cpu)
        self.net.eval()

        print('Finished loading model!')
        print(f'Using device: {self.device}')

    def load_model(self, model, pretrained_path, cpu=False):
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"权重文件不存在: {pretrained_path}")

        if cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda())

        model.load_state_dict(pretrained_dict, strict=False)
        model = model.to(self.device)
        return model

    def detect(self, img_path, confidence_threshold=0.02, nms_threshold=0.4):
        """
        检测单张图片中的人脸
        Args:
            img_path: 图片路径
            confidence_threshold: 置信度阈值
            nms_threshold: NMS阈值
        Returns:
            dets: 检测结果 [x1, y1, x2, y2, score, landm1_x, landm1_y, ...]
            img: 原图像
        """
        # 读取图片
        img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_raw is None:
            raise ValueError(f"无法读取图片: {img_path}")

        img = np.float32(img_raw)

        # 图像标准化
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        # 前向传播
        loc, conf, landms = self.net(img)

        # 解码检测结果
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data

        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()


        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # 忽略低分检测
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # 保持至少一个检测结果
        dets = np.concatenate((dets, landms), axis=1)

        return dets, img_raw

    def draw_detections(self, image, dets, save_path=None, show=True):
        """
        Args:
            image: 原图像
            dets: 检测结果
            save_path: 保存路径
            show: 是否显示结果
        """
        img = image.copy()

        for detection in dets:
            # 边界框坐标和置信度
            x1, y1, x2, y2, score = detection[:5]

            # 只绘制高置信度检测
            if score < 0.5:
                continue

            # 绘制边界框
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # 绘制置信度
            text = f'{score:.2f}'
            cv2.putText(img, text, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 绘制关键点（如果有）
            if len(detection) > 5:
                landmarks = detection[5:].reshape(5, 2)
                for i, (x, y) in enumerate(landmarks):
                    print(f"x{i+1}:{x:.2f},y{i+1}:{y:.2f}")
                    cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 2)

        # 显示或保存结果
        if show:
            cv2.imshow('Face Detection', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_path:
            cv2.imwrite(save_path, img)
            print(f"结果已保存到: {save_path}")

        return img


def main():
    """主函数：检测单张照片"""
    # 初始化检测器
    detector = FaceDetector(
        network='mobile0.25',
        trained_model='D:/pywork/machine_vision/machine_vision_0/Pytorch_Retinaface/weights/mobilenet0.25_Final.pth',
        cpu=True  # 如果有GPU可以设为False
    )

    # 图片路径（修改为你的图片路径）
    image_path = "D:/pywork/machine_vision/machine_vision_0/Pytorch_Retinaface/for_face_detect.jpg"  # 替换为你的图片路径

    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}")
        print("请将图片放在项目根目录下，或修改image_path变量")
        return

    try:
        # 进行人脸检测
        print("开始人脸检测...")
        start_time = time.time()

        dets, img_raw = detector.detect(image_path)

        end_time = time.time()
        print(f"检测完成！耗时: {end_time - start_time:.2f}秒")
        print(f"检测到 {len(dets)} 张人脸")

        # 绘制并显示结果
        result_image = detector.draw_detections(
            img_raw,
            dets,
            save_path="detection_result.jpg",
            show=True
        )

    except Exception as e:
        print(f"检测过程中出现错误: {e}")


if __name__ == "__main__":
    main()