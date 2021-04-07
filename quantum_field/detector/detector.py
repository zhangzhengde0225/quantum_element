import os,sys
import yaml
import torch
import random
import cv2
import numpy as np
from detector.YOLOv5.utils.general import non_max_suppression, scale_coords, plot_one_box


class Detector(object):
	def __init__(self, detector_type='SEYOLOv5', cfg_file=None):
		self.detecotr_type = detector_type
		self.cfg_file = cfg_file if cfg_file is not None else f'{os.getcwd()}/detector/config_files/{detector_type}_config_file.yaml'
		self.model, self.device, self.cfg, self.half = self.init_detector(detector_type)
		self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
		self.filt_classes = [int(x) for x in self.cfg['filt_classes'].split(',')]
		self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]  # 最大100种颜色，超过的重新取

	def init_detector(self, dettype):
		"""load model, load weights"""
		print(f'Detector type: {dettype}. ', end='')
		if dettype == 'SEYOLOv5':
			return self.assemble_yolov5()
		else:
			raise NameError(f'unsupported detector type: {dettype}')

	def assemble_yolov5(self):
		sys.path.append(f"{os.getcwd()}/detector/YOLOv5")
		from models.experimental import attempt_load
		from utils.torch_utils import select_device

		# read cfg_file
		with open(self.cfg_file, 'r') as f:
			cfg = yaml.load(f, Loader=yaml.FullLoader)
		device = select_device(cfg['device'])

		model = attempt_load(cfg['weights'], map_location=device)
		# model = torch.hub.load('ultralytics/yolov5', cfg['weights'], pretrained=True)
		sys.path.remove(f"{os.getcwd()}/detector/YOLOv5")

		half = cfg['half'] and device.type != 'cpu'
		if half:
			model.half()

		return model, device, cfg, half

	def run_through_once(self, imgsz):
		img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
		_ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once

	def detect(self, path, img, im0s):
		device = self.device
		model = self.model
		half = self.half
		cfg = self.cfg

		paths = [path] if isinstance(path, str) else path
		im0s = im0s[np.newaxis, :] if im0s.ndim == 3 else im0s
		imgs = img[np.newaxis, :] if img.ndim == 3 else img
		imgs = torch.from_numpy(imgs).to(device)
		imgs = imgs.half() if half else imgs.float()
		imgs /= 255.0

		# inference
		model.eval()
		pred = model(imgs, augment=cfg['augment'])[0]
		# NMS
		pred = non_max_suppression(
			pred, cfg['conf_thres'], cfg['iou_thres'], classes=0, agnostic=cfg['agnostic_nms'])

		# 处理结果： 转为nparray
		assert len(pred) == 1  # 只允许单图检测
		ret = []
		for i, det in enumerate(pred):
			p, im0 = paths[i], im0s[i]
			if det is not None and len(det):
				# print(det.shape, img.shape)
				det[:, :4] = scale_coords(imgs.shape[2:], det[:, :4], im0.shape).round()  # 有时候会返回None
				det = det.cpu().numpy()
			ret.append(det) if det is not None else None

		ret = np.array(ret)  # [1, num_obj, 6], None
		# print(ret.shape, ret)
		if cfg['filt_classes'] is not None and len(ret) > 0:  # filter class
			valid_ret = []
			# print(ret.shape, ret, len(ret))
			for valid_cls in cfg['filt_classes'].split(','):
				tmp = ret[ret[:, :, -1] == int(valid_cls)]
				if len(tmp) != 0:
					valid_ret.append(tmp) if len(valid_ret) == 0 else valid_ret.extend(tmp)
					# valid_ret.extend(tmp) if len(tmp) != 0 else None
			ret = np.array(valid_ret)
			# ret = ret[np.newaxis, :, :]
		# print('xx', ret.shape, ret)
		ret = ret[0, :, :] if len(ret) > 0 else None

		return ret  # nparray, [num_obj, 6] 6: xyxy,conf,cls

	def imshow(self, im0s, ret):
		# print(im0s.shape, ret.shape)
		for *xyxy, conf, cls in ret:
			label = f'{self.names[int(cls)]} {conf:.2f}'
			plot_one_box(xyxy, im0s, label=label, color=self.colors[int(cls)], line_thickness=3)
		cv2.imshow('xx.png', im0s)
		cv2.waitKeyEx(0)
