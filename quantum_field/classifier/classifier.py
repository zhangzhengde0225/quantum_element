"""
status classifier
"""
import os
from pathlib import Path
from easydict import EasyDict
import yaml
import torch
import numpy as np
import damei as dm
import copy
import cv2
import random
from classifier.feature_builder import FeatureBuilder


class Classifier(object):
	def __init__(self, classifier_type='ResNet50', cfg_file=None):
		self.classifier_type = classifier_type
		self.cfg_file = cfg_file if cfg_file is not None else f'{os.getcwd()}/classifier/config_files/{classifier_type}_config_file.yaml'
		self.model, self.cfg, self.device = self.init_classifier(classifier_type)
		self.feature_builder = FeatureBuilder(cfg_file=self.cfg_file)
		self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]  # 最大100种颜色，超过的重新取

	def init_classifier(self, clsitype):
		print(f'Classifier type: {clsitype}. ', end='')
		if clsitype == 'Resnet50':
			return self.assemble_Resnet50()
		else:
			raise NameError(f'unsupported classifier type: {clsitype}')

	def assemble_Resnet50(self):
		from classifier.ResNet50.utils import general as resnet_general
		from classifier.ResNet50.models import resnet_zoo_behavior
		with open(self.cfg_file, 'r') as f:
			cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
		device = dm.torch_utils.select_device(cfg['device'])
		model = resnet_zoo_behavior.attempt_load(
			model_name='resnet50', pretrained=False, num_classes=cfg['num_classes'])
		model = resnet_general.load_weights(
			weights_path=cfg['weights_path'], model=model, need_ckpt=False, device=device)

		if len(cfg['device'].split(',')) >= 2:
			model.to(device)
			model = torch.nn.DataParallel(model, device_ids=cfg['device']).to(device)
		else:
			model.to(device)

		return model, cfg, device

	def build_feature(self, poser_ret, img_name, **kwargs):
		return self.feature_builder.build_feature(poser_ret, img_name, kwargs)

	def build_and_save_feature(self, poser_ret, path, save_dir=''):
		"""
		:param meta: [num_obj, 3], 3: img_name, bbox, tid
		:param features: [num_obj, h, c*w, 3]
		:return:
		"""
		meta, features = self.build_feature(poser_ret, path)
		num_person = len(features) if features is not None else 0
		print(f'num_person: {num_person} {path}', end='')
		self.feature_builder.save_feature(meta, features, save_dir=save_dir)

	def detect_status(self, poser_ret, path, **kwargs):
		"""

		:param poser_ret: [num_obj, 10]
		:param path: img path
		:return: [num_obj, 12]
		"""
		# print(poser_ret.shape, path)

		meta, features = self.build_feature(poser_ret, path, **kwargs)

		if features is None:
			return None
		model = self.model
		cfg = self.cfg
		device = self.device
		batch_size = cfg['batch_size']

		features = np.array(features, dtype=np.float32)
		features = features/255.0
		features = features.transpose(0, 3, 1, 2)

		# print(features.shape, type(features), features.dtype)  # [num_cls, 192, 640, 3] n, h, w, 3, ndarray
		# 补全
		fs = features.shape  # features shape
		# remainder = fs[0] % batch_size  # 3, or 0, 1, 2, 3, 4, 5, 6, 7
		# if remainder != 0:
		# 	addend = np.zeros((batch_size-remainder, fs[1], fs[2], fs[3]), dtype=np.uint8)
		# 	features = np.concatenate((features, addend), axis=0)
		leftover = 1 if fs[0] % batch_size else 0
		num_batches = fs[0] // batch_size + leftover

		model.eval()
		features = torch.from_numpy(copy.deepcopy(features)).contiguous()
		# features = features.permute(0, 3, 1, 2)
		features = features.to(device)
		outputs = []
		for i in range(num_batches):
			bfeature = features[i*batch_size:min((i+1)*batch_size, len(features)), ...]
			output = model(bfeature)  # [batch_size, 7]
			output = torch.argmax(output, dim=1).cpu().numpy().tolist()  # [batch_size, ]
			outputs.extend(output)

		# 补全outputs, 原因是可能会出现跟踪到了，但是没有关节点的问题，导致也无法检测状态
		complete_outputs = [None] * len(poser_ret)
		outputs_dict = dict(zip([x[5] for x in meta], outputs))
		for i in range(len(poser_ret)):
			tid = poser_ret[i][5]
			cls = outputs_dict[tid] if tid in outputs_dict.keys() else None
			complete_outputs[i] = [cls, model.classes[cls]] if cls is not None else [None, None]

		assert len(complete_outputs) == len(poser_ret)

		clser_ret = np.array((complete_outputs), dtype=object)
		final_clser_ret = np.concatenate((poser_ret, clser_ret), axis=1)

		return final_clser_ret  # [num_obj, ]

	def imshow(self, orig_img, result, show_name='xxx', resize=None):
		from butils import general

		out_img = np.copy(orig_img)
		if result is None:
			pass
		else:
			# result  [num_obj, 10]，10: xyxy cls tid trace keypoints kp_score proposal_score
			for i, person in enumerate(result):
				bbox_xyxy = person[:4]
				tid = person[5]
				trace = person[6]
				keypoints = person[7]
				kp_score = person[8]
				status = person[11] if person[11] is not None else 'unk'
				out_img = general.plot_one_box_trace_pose_status(
					bbox_xyxy, out_img,
					label=f"{tid:<2}", color=self.colors[tid % len(self.colors)], trace=trace, status=status,
					keypoints=keypoints, kp_score=kp_score, skeleton_thickness=2)

		if resize is not None:
			out_img = cv2.resize(out_img, resize)
		cv2.imshow(show_name, out_img)
		if cv2.waitKey(5) == ord('q'):  # q to quit
			raise StopIteration


