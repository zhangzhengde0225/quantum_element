import os
import yaml
from easydict import EasyDict
import torch
import numpy as np
import cv2
import random

import damei as dm
from detector.YOLOv5.utils.torch_utils import select_device
from poser.utils import test_transform
from alphapose.utils.pPose_nms import pose_nms
from alphapose.utils.transforms import get_func_heatmap_to_coord


class Poser(object):
	def __init__(self, poser_type='AlphaPose', cfg_file=None):
		self.poser_type = poser_type
		self.cfg_file = cfg_file if cfg_file is not None else f'{os.getcwd()}/poser/config_files/{poser_type}_config_file.yaml'
		self.model, self.cfg, self.device = self.init_poser(poser_type)
		self.heatmap_to_coord = get_func_heatmap_to_coord(self.cfg)
		self.eval_joints = [*range(0, self.cfg.DATA_PRESET.NUM_JOINTS)]
		self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]  # 最大100种颜色，超过的重新取

	def init_poser(self, postype):
		print(f'Poser type: {postype}. ', end='')
		if postype == 'AlphaPose':
			return self.assemble_AlphaPose()
		else:
			raise NameError(f'unsupported poser type: {postype}')

	def assemble_AlphaPose(self):
		from alphapose.models import builder

		with open(self.cfg_file, 'r') as f:
			cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
		device = select_device(cfg['device'])
		model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
		print(f'Loading weights from {cfg["weights"]}')
		model.load_state_dict(torch.load(cfg['weights'], map_location=device))
		# print(cfg.device, device)
		if len(cfg['device'].split(',')) >= 2:
			model.to(device)
			model = torch.nn.DataParallel(model, device_ids=cfg['device']).to(device)
		else:
			model.to(device)

		return model, cfg, device

	def detect_pose(self, track_ret, raw_img, img_name, return_type='AlphaPose'):
		"""

		:param track_ret: np.array(object): [num_obj, 10]，10: xyxy cls tid trace
		:param raw_img: np.array [h, w, 3]
		:param img_name: abs path to the image
		:param return_type:
				AlphaPose: list, 长度num_obj, 每个元素是字典，里面有img_name, result: keypoints, kp_score, proposcal_score, id, box, box是x1y1wh格式
				zzd: np.array(object): [num_obj, 10]，10: xyxy cls tid trace keypoints kp_score proposal_score
		:return: list or np.array(object)
		"""
		if track_ret is None:
			return None
		model = self.model
		cfg = self.cfg
		device = self.device

		features = np.zeros((len(track_ret), 3, *cfg.DATA_PRESET.IMAGE_SIZE), dtype=np.float32)  # [5, 3, 256, 192]
		cropped_bboxes = np.zeros((len(track_ret), 4), dtype=np.float32)
		bboxes = np.array(track_ret[:, :4], dtype=np.int32)
		for i in range(len(track_ret)):
			features[i], cropped_bbox = test_transform.test_transform(raw_img, bboxes[i], cfg.DATA_PRESET.IMAGE_SIZE)
			cropped_bboxes[i] = torch.FloatTensor(cropped_bbox)
		# print(features[0].shape)  # [3, 256, 192]
		batch_size = cfg['feature_batch_size']
		leftover = 1 if len(features) % batch_size else 0
		num_batches = len(features)//batch_size + leftover

		features = torch.from_numpy(features)
		features = features.to(device)
		model.eval()
		human_pose = []
		for i in range(num_batches):
			feature = features[i * batch_size:min((i+1)*batch_size, len(features))]
			hm_i = model(feature)
			human_pose.append(hm_i)
		human_pose = torch.cat(human_pose)  # [num_person, 136, 64, 48]
		human_pose = human_pose.detach().cpu()

		# postprocess
		# scores = track_ret[:, 4:5]
		pose_coords = []
		pose_scores = []
		assert len(human_pose) == len(track_ret)
		for i in range(len(human_pose)):  # every person
			crop_bbox = cropped_bboxes[i].tolist()
			pose_coord, pose_score = self.heatmap_to_coord(
				human_pose[i][self.eval_joints], crop_bbox,
				hm_shape=cfg.DATA_PRESET.HEATMAP_SIZE, norm_type=cfg.LOSS.NORM_TYPE)
			# print(pose_coord.shape, pose_score.shape)  # ndarray [136, 2], ndarray [136, 1]
			pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
			pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
		preds_coords = torch.cat(pose_coords)
		preds_scores = torch.cat(pose_scores)

		scores = np.ones((len(track_ret), 1))  # 其实应该传入目标检测的score才对
		ids = np.array(track_ret[:, 5:6], dtype=np.int32)
		boxes, scores, ids, preds_img, preds_scores, pick_ids = pose_nms(
			torch.from_numpy(bboxes), torch.from_numpy(scores), torch.from_numpy(ids),
			preds_coords, preds_scores, cfg['min_box_area'])
		# print(type(ret), len(ret))  # list, len:6, 每个元素也是list, len：4 4是目标个数，里面是每个目标的bbox
		if return_type == 'AlphaPose':
			_result = []
			for k in range(len(scores)):
				_result.append(
					{
					'keypoints': preds_img[k],
					'kp_score': preds_scores[k],
					'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
					'idx': ids[k],
					'box': [boxes[k][0], boxes[k][1], boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1]]
					}
				)
			result = {
				'imgname': img_name,
				'result': _result
			}
			return result
		elif return_type == 'zzd':
			# print(track_ret)

			align_idxs = {}  # pose的结果和track根据bbox的iou对齐，键：track的idx，值：poser的idx
			for k in range(len(scores)):
				pose_box = np.array(boxes[k], dtype=np.int32)
				ious = [dm.general.bbox_iou(pose_box, np.array(x[:4], dtype=np.int32), return_np=True) for x in track_ret]
				ious = np.array(ious)
				idx = np.argmax(ious)
				assert ious[idx] > cfg['iou_thresh_for_track_pose_align']
				assert idx not in align_idxs.keys()
				align_idxs[idx] = k

			# print(align_idxs)
			poser_ret = [None] * len(track_ret)
			for z in range(len(track_ret)):
				if z not in align_idxs.keys():
					keypoints, kp_score, proposal_score = None, None, None
				else:
					poser_idx = align_idxs[z]
					keypoints = preds_img[poser_idx]
					kp_score = preds_scores[poser_idx]
					proposal_score = torch.mean(preds_scores[poser_idx] + scores[poser_idx] + 1.25 * max(preds_scores[poser_idx]))
					keypoints = keypoints.numpy().tolist()
					kp_score = kp_score.numpy().tolist()
					proposal_score = float(proposal_score.numpy())
				poser_ret[z] = [keypoints, kp_score, proposal_score]
			poser_ret = np.array(poser_ret, dtype=object)
			final_poser_ret = np.concatenate((track_ret, poser_ret), axis=1)
			# print(final_poser_ret)

			return final_poser_ret

		else:
			raise NameError(f'not supported return type: {return_type}')

	def imshow(self, orig_img, result, show_name='xxx', resize=None):
		from butils import general

		# result  [num_obj, 10]，10: xyxy cls tid trace keypoints kp_score proposal_score
		out_img = np.copy(orig_img)
		if result is None:
			pass
		else:
			for i, person in enumerate(result):
				bbox_xyxy = person[:4]
				tid = person[5]
				trace = person[6]
				status = ''
				keypoints = person[7]
				kp_score = person[8]
				# print(type(keypoints), keypoints)  # list
				# print(type(kp_score), kp_score)  # list
				out_img = general.plot_one_box_trace_pose_status(
					bbox_xyxy, out_img,
					label=f"{tid:<2}", color=self.colors[tid % len(self.colors)], trace=trace, status=status,
					keypoints=keypoints, kp_score=kp_score, skeleton_thickness=3, kp_radius=5)
		if resize:
			out_img = cv2.resize(out_img, resize)
		cv2.imshow(show_name, out_img)
		if cv2.waitKey(5) == ord('q'):  # q to quit
			raise StopIteration

