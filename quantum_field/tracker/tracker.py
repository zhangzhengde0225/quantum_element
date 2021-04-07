"""
跟踪加载器，目前支持Deepsort
"""

import os
import numpy as np
import yaml
import cv2
import torch
from detector.YOLOv5.utils.general import plot_one_box_track_status
import random
import damei as dm
import matplotlib.pyplot as plt


class Tracker(object):
	def __init__(self, tracker_type='Deepsort', cfg_file=None):
		self.detecotr_type = tracker_type
		self.cfg_file = cfg_file if cfg_file is not None else f'{os.getcwd()}/tracker/config_files/{tracker_type}_config_file.yaml'
		self.model, self.cfg = self.init_tracker(tracker_type)
		self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]  # 最大100种颜色，超过的重新取

	def init_tracker(self, tratype):
		print(f'Tracker type: {tratype}. ', end='')
		if tratype == 'Deepsort':
			return self.assemble_deepsort()
		else:
			raise NameError(f'unsupported tracker: {tratype}')

	def assemble_deepsort(self):

		from tracker.deep_sort import DeepSort

		with open(self.cfg_file, 'r') as f:
			cfg = yaml.load(f, Loader=yaml.FullLoader)

		model = DeepSort(cfg['weights'], use_cuda=cfg['use_cuda'])

		return model, cfg

	def track(self, det_ret, img):
		"""
		传入该图和这张图的目标检测结果，获取tid和轨迹
		:param input:
		:return:
		"""
		if det_ret is None:  # 画面中没有检测到目标
			return 
		bbox_tracking = dm.general.xyxy2xywh(det_ret[:, :4])
		cls_conf = det_ret[:, 4]
		cls_ids_tracking = det_ret[:, 5]
		ret = self.model.update(bbox_tracking, cls_conf, cls_ids_tracking, img, return_xywh=False)  # 返回xyxy
		# 返回的ret是np.array, [18, 7], 18个目标，7: xyxy cls tid trace 少了conf
		if len(ret) == 0:  # no tracking result, use object detect result
			ret = np.array([np.array(x[:4], dtype=np.int32).tolist()+[int(x[5])]+[0]+[[[0, 0], [0, 0]]] for x in det_ret], dtype=object)
		return ret

	def imshow(self, raw_img, ret, show_name='xxx'):
		out_img = raw_img
		for x in ret:
			bbox_xyxy = x[:4]
			cls = x[4]
			tid = x[5]
			trace = x[6]
			out_img = plot_one_box_track_status(
				bbox_xyxy, out_img, label=f"{tid:<2}", color=self.colors[tid], trace=trace, status='0')
		# exit()
		cv2.imshow(show_name, out_img)
		if cv2.waitKey(5) == ord('q'):  # q to quit
			raise StopIteration
		# cv2.waitKey(0)
		# if cv2.waitKey(1) == ord('q'):  # q to quit
		# 	raise StopIteration


