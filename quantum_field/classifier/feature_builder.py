"""
根据已有的img_name cls bbox id trace keypoints kpscore信息，构建用于分类状态的特征，保存为数据集
"""
import os
import yaml
from easydict import EasyDict
from pathlib import Path
import damei as dm
import numpy as np
import copy
import cv2
import shutil
from butils import general


class FeatureBuilder(object):
	def __init__(self, cfg_file):
		self.cfg = self.read_cfg(cfg_file)
		self.c = self.cfg['consecutive_frame']
		self.frames_data = [None] * self.c

	def read_cfg(self, cfg_file):
		with open(cfg_file, 'r') as f:
			cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
		return cfg

	def save_feature(self, meta, features, save_dir=''):
		"""
		保存特征
		"""
		# print(len(meta), features.shape, meta[0][0])
		# print(type(meta), type(features))
		if features is None:
			return

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		for i, (p, bbox, tid) in enumerate(meta):  # img_path, bbox, tid
			# 读取目标状态
			stem = Path(p).stem
			label_name = f"{Path(p.replace('raw_images', 'labels')).parent}/track_{stem}.txt"
			if not os.path.exists(label_name):  # 不存在标注文件直接下一张
				print(f'\nWARNING, label file not exists: {label_name}')
				break
			with open(label_name, 'r') as f:
				label_data = f.readlines()
			im0 = cv2.imread(p)
			status = self.read_status(bbox, label_data, im0)  # ['0', 'walking']
			if status is None:
				continue
			status = status[-1]
			if status == 'unknown':
				continue
			feature = features[i]
			feature_path = f'{save_dir}/feature_{status}_{tid:0>6}_{stem}.jpg'
			cv2.imwrite(feature_path, feature)
			# print(p, bbox, tid, feature.shape, status)

	def read_status(self, bbox, label_data, im0, iou_thresh=0.85):
		# 逻辑，计算出bbox与label中所有标注的IOU，IOU最大的且大于阈值0.85，再看tid
		# bbox = dm.general.xywh2xyxy(bbox[np.newaxis, :], need_scale=False)
		ious = [None] * len(label_data)
		for i, line in enumerate(label_data):
			bbox_label = line.split('bbox:')[-1].split('trace:')[0].split()  # x1y1x2
			bbox_label = np.array(bbox_label, dtype=np.float32)  # 标注的bbox是
			bbox_label = dm.general.xywh2xyxy(bbox_label[np.newaxis, :], need_scale=True, im0=im0)
			iou = dm.general.bbox_iou(
				bbox, bbox_label, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, return_np=True)
			# print(bbox_label, bbox, tid, type(tid), iou)
			ious[i] = float(iou)
		ious = np.array(ious)
		idx = np.argmax(ious)
		if not ious[idx] > iou_thresh:
			return None
		else:
			status = label_data[idx].split('status:')[-1].split() if 'status:' in label_data[idx] else None
			return status

	def build_feature(self, ret, img_name, is_camera=False):
		"""
		创建特征，第一步，读取和更新信息，第二步
		:param ret: np.array(object): [num_obj, 10]，10: xyxy cls tid trace keypoints kp_score proposal_score
		:return: meta: cname, xyxy, ctid, [num_obj, h, w*c, 3], wh是配置文件制定的特征宽和高，c是连续帧数目
		"""
		if ret is None:
			return None, None
		# 给数据
		frdata = self.frames_data
		c = self.cfg['consecutive_frame']
		frdata[:c - 1] = frdata[1:c]
		frdata[-1] = ret
		self.frames_data = frdata

		# 处理frdata，针对当前帧的每一个tid，构建完备的数据:[c, 5]
		c_data = frdata[-1]  # current data [num_obj, 10]
		features_meta = []
		features = []
		for i, d in enumerate(c_data):
			bbox, tid, kps, kpss = np.array(d[:4], dtype=np.int32), d[5], d[7], d[8]

			# 获取完备的数据[c, 5], 5: frame_name, tid, bbox, kps, kpss
			inter_data = [[img_name, bbox, tid, kps, kpss]]
			tmpname, tmpbbox, tmptid = copy.deepcopy(img_name), copy.deepcopy(bbox), copy.deepcopy(tid)
			tmpkps, tmpkpss = copy.deepcopy(kps), copy.deepcopy(kpss)
			for j in reversed(range(c-1)):
				e = frdata[j]  # [num_obj, 10]
				if e is None:  # 无数据，继承tmp
					pname, pbbox, ptid, pkps, pkpss = tmpname, tmpbbox, tmptid, tmpkps,tmpkpss  # pre frame result
				else:
					tmpee = [[np.array(x[:4], dtype=np.int32), x[5], x[7], x[8]] for x in e if x[5] == tid]
					if len(tmpee) > 1 and not (tid == 0):
						raise NameError(f'tmpee, {len(tmpee)} {tmpee[0]}')
					if len(tmpee) == 0:  # 有数据，但没有相同的tid，也是继承
						pname, pbbox, ptid, pkps, pkpss = tmpname, tmpbbox, tmptid, tmpkps, tmpkpss
					else:  # 有数据，有相同的tid，再看
						ebbox, etid, ekps, ekpss = tmpee[0]
						if ekps is None:  # 有数据，有相同的tid，无kpoints，也是继承
							pname, pbbox, ptid, pkps, pkpss = tmpname, tmpbbox, tmptid, tmpkps, tmpkpss
						else:
							pbbox, ptid, pkps, pkpss = tmpee[0]
							imp = Path(img_name)  # img_path
							new_stem = int(imp.stem) - ((c-1)-j)
							pname = f'{imp.parent}/{new_stem:0>6}{imp.suffix}'
							tmpname = pname
							tmpbbox, tmptid, tmpkps, tmpkpss = tmpee[0]
							if not is_camera:
								assert os.path.exists(pname)
				inter_data.append([pname, pbbox, ptid, pkps, pkpss])

			# 根据完备的[c, 5]中间数据，构建特征图，v1格式为[h, w*c, 3], c帧的图像横向合并在一起
			cname, cbbox, ctid, cfeature = self.gen_feature(inter_data)  # current image_path, bbox, tid and feature
			if cfeature is not None:
				features_meta.append([cname, cbbox[0], cbbox[1], cbbox[2], cbbox[3], ctid])
				features.append(cfeature)
		features = np.array(features)
		if len(features) == 0:
			return None, None
		else:
			return features_meta, features  # [num_obj, h, w*c, 3], wh是配置文件制定的特征宽和高，c是连续帧数目

	def gen_feature(self, inter_data):
		"""
		根据输入的数据从图像从裁剪数据产生特征图
		:param inter_data: list, img_name, bbox, tid, kps, kpss
		:return: feature, [w, h, 3*c]，c张图像拼接而成，每张图像是人体关键点绘制的，wh是裁剪再resize的图像
		"""
		feature_size = self.cfg['feature_size']
		kp_radius = self.cfg['keypoints_radius']
		skeleton_thickness = self.cfg['skeleton_thickness']
		current_imgname = inter_data[0][0]
		current_bbox = inter_data[0][1]
		current_tid = inter_data[0][2]
		if inter_data[0][3] is None:  # 如果当前帧并没有keypoints，返回None
			return current_imgname, current_bbox, current_tid, None
		bboxes = np.array([x[1] for x in inter_data])
		convas_bbox = np.concatenate((np.min(bboxes[:, :2], axis=0), np.max(bboxes[:, 2:], axis=0)))
		convas = np.zeros([convas_bbox[3]-convas_bbox[1], convas_bbox[2]-convas_bbox[0], 3], np.uint8)
		convas[...] = 114
		cvs, ratio, (dw, dh), recover = general.letterbox(
			convas, new_shape=feature_size, auto=False, scaleFill=False, scaleup=True, roi=None)
		# print(bboxes.shape, bboxes, convas.shape, cvs.shape, convas_bbox)
		# print(inter_data[0][2], ratio, dw, dh, recover)
		# draw keypoints and skeleton 把keypoints的坐标从全图坐标转换为convas坐标，并转换为resize之后的坐标
		# new_x = ratio_w * [x - (x1 of convas)] + dw
		kps = np.array([x[3] for x in inter_data])  # [c, 136, 2]
		kpsx = ratio[1] * (kps[:, :, 0] - convas_bbox[0]) + dw
		kpsy = ratio[0] * (kps[:, :, 1] - convas_bbox[1]) + dh
		cvs_kps = np.concatenate((kpsx[:, :, np.newaxis], kpsy[:, :, np.newaxis]), axis=2)
		# print(type(kps), kps.shape, kps[0, :2, :2], cvs_kps.shape, cvs_kps[0, :2, :2])

		# feature = np.zeros((len(inter_data), feature_size[0], feature_size[1], 3))
		feature = [None] * len(inter_data)
		for i in range(len(inter_data)):
			use_raw_image_as_convas = self.cfg['use_raw_image_rather_grayscale_as_convas']
			if use_raw_image_as_convas:
				img = cv2.imread(inter_data[i][0])
				convas = img[convas_bbox[1]:convas_bbox[3], convas_bbox[0]:convas_bbox[2]]
				cvs, ratio, (dw, dh), recover = general.letterbox(
					convas, new_shape=feature_size, auto=False, scaleFill=False, scaleup=True, roi=None)
			kps = cvs_kps[i]  # [136, 2]
			kpss = np.array(inter_data[i][4])  # [136, 1]
			# 画关键点和骨架， 传入prev_kps绘制骨骼运动
			prev_kps = cvs_kps[i+1] if i < (len(inter_data)-1) else None
			prev_kpss = np.array(inter_data[i+1][4]) if i < (len(inter_data)-1) else None
			drawed_cvs = general.draw_keypoints_and_skeleton(
				copy.deepcopy(cvs), kps, kpss,
				kp_radius=kp_radius, skeleton_thickness=skeleton_thickness,
				prev_keypoints=prev_kps, prev_kp_score=prev_kpss)

			feature[i] = drawed_cvs
			show_single_feature = False
			if show_single_feature and i == 0:
				feature = cvs
				a = inter_data[i]
				cv2.imshow(f'{Path(a[0]).stem}_x{i}{a[2]}', drawed_cvs)
				if cv2.waitKey(0) == ord('q'):
					raise StopIteration
		feature = np.array(feature)
		new_feature = np.zeros((feature_size[0], feature_size[1]*len(inter_data), 3), dtype=np.uint8)  # [192, 128*5, 3]
		for i, x in enumerate(feature):
			new_feature[:, i*feature_size[1]:(i+1)*feature_size[1], :] = x
		feature = new_feature

		show_feature = False
		if show_feature:
			cv2.imshow(f'{Path(current_imgname).stem}_tid_{current_tid}', feature)
			if cv2.waitKey(0) == ord('q'):
				raise StopIteration
		return current_imgname, current_bbox, current_tid, feature
