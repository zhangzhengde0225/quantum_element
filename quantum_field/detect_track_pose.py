"""
检测、跟踪、姿态，构建状态分类特征
"""
import os, sys
import shutil
from pathlib import Path
import argparse
from detector.detector import Detector
from dataloader.dataloader import Dataloader
from tracker.tracker import Tracker
from poser.poser import Poser
from classifier.classifier import Classifier
import cv2


def run(opt):

	# output dir
	if os.path.exists(opt.save_dir):
		shutil.rmtree(opt.save_dir)
	os.makedirs(opt.save_dir)

	# load dataset
	dataset = Dataloader(source=opt.source, imgsz=opt.img_size).dataset

	# load object detection model, and weights
	detector = Detector(detector_type=opt.detector_type, cfg_file=opt.detector_cfg_file)
	detector.run_through_once(opt.img_size)  # 空跑一次

	# load object tracking model
	tracker = Tracker(tracker_type=opt.tracker_type, cfg_file=opt.tracker_cfg_file)

	# load pose detection model
	poser = Poser(poser_type=opt.poser_type, cfg_file=opt.poser_cfg_file)

	# load classifier model
	clssifier = Classifier(classifier_type=opt.classifier_type, cfg_file=opt.classifier_cfg_file)

	print(detector.device, detector.cfg)
	filt_with_txt = False  # 先分析一下status标注文件.txt，存在的才进行检测，这样能加快速度
	if filt_with_txt:
		from classifier.data_analyse import anaylise_label
		label_ret = anaylise_label()
		label_stems = [x[0] for x in label_ret]

	for img_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
		# print(type(img), type(im0s))
		# print(type(im0s), im0s.shape)
		if dataset.is_camera:
			im0s = im0s[0]
			path = f'{path[0]}/{img_idx:0<6}.jpg'
		if filt_with_txt:
			fold_stem = path.split('/')[-2]
			idx = label_stems.index(fold_stem)
			# print(fold_stem, label_stems, idx)
			img_stem = Path(path).stem
			valid_stems = [Path(x).stem for x in label_ret[idx][-1]]
			in_it = f'track_{img_stem}' in valid_stems
			# print(path, in_it, label_ret[idx][-1][0])
			if not in_it:
				continue
		# img: [3, w, h], preprocess, inference, NMS,
		det_ret = detector.detect(path, img, im0s)  # detect result: nparray, [num_obj, 6] 6: xyxy,conf,cls
		# detector.imshow(im0s, det_ret)
		# track
		tra_ret = tracker.track(det_ret, im0s)  # track result: list, [num_obj, 7], 7: xyxy, cls, tid, trace
		# print(tra_ret[:, 5])
		# tracker.imshow(im0s, tra_ret, path)
		# pose detect
		pose_ret = poser.detect_pose(tra_ret, im0s, path, return_type='zzd')
		# zzd format: np.array(object): [num_obj, 10]，10: xyxy cls tid trace keypoints kp_score proposal_score
		# print(pose_ret)
		poser.imshow(im0s, pose_ret, path, resize=(1280, 720))
		# classifier
		if opt.feature_save_dir is not None:  # 保存特征的
			clssifier.build_and_save_feature(pose_ret, path, save_dir=opt.feature_save_dir)
			print(f'\rsaving features: [{img_idx + 1:>3}/{len(dataset)}] ', end='')
			continue

		# status_ret = clssifier.detect_status(pose_ret, path, is_camera=dataset.is_camera)
		# zzd format: np.array(object): [num_obj, 12], 12: 比10多了status_idx和status
		# clssifier.imshow(im0s, status_ret, show_name='x', resize=(1280, 720))
		# print(status_ret)

		if img_idx == 10:
			if cv2.waitKeyEx(0) == ord('q'):
				raise StopIteration


def arg_set():
	parser = argparse.ArgumentParser()
	parser.add_argument('--source', type=str, default='',
						help='source: file, folder, 0 for webcam')
	parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
	parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels) for dataloader')

	parser.add_argument('--detector_type', type=str, default='SEYOLOv5', help='module for object detection')
	parser.add_argument('--detector_cfg_file', type=str, default=None,
						help='config file for assemble detection model, use DEFAULT cfg in detecotr/config_files if None')

	parser.add_argument('--tracker_type', type=str, default='Deepsort', help='module for object tracking')
	parser.add_argument('--tracker_cfg_file', type=str, default=None,
						help='config file for assemble tracking model, use DEFAULT cfg in tracker/config_files if None')

	parser.add_argument('--poser_type', type=str, default='AlphaPose', help='module for human pose detection')
	parser.add_argument('--poser_cfg_file', type=str, default=None,
						help='config file for assemble poser model, use DEFAULT cfg in poser/config_files if None')

	parser.add_argument('--classifier_type', type=str, default='Resnet50', help='module for human pose detection')
	parser.add_argument('--classifier_cfg_file', type=str, default=None,
						help='config file for assemble classifier model, use DEFAULT cfg in classifier/config_files if None')

	parser.add_argument('--build_classifier_feature', default=True,
						help='use detector tracker poser to build feature for status classifier')

	parser.add_argument('--feature_save_dir', default=None,
						help='dir to save feature, if not None, save feature only')

	parser.add_argument('--save-img', type=bool, default=False, help='save img or not')
	parser.add_argument('--view-img', action='store_true', help='display results')
	parser.add_argument('--save-txt', type=bool, default=False, help='save results to *.txt')
	parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')

	parser.add_argument('--classfier_weights', type=str, default='', help='path to rtatus classfier weights of resnet')
	parser.add_argument('--continous', type=int, default=5, help='continous frames')
	parser.add_argument('--save-sc-txt', type=bool, default=True,
						help='save status classfication txt or not, sc_stem.txt')
	opt = parser.parse_args()
	return opt


if __name__ == '__main__':
	opt = arg_set()

	user, pwd, ip, channel = "admin", "123qweasd", "192.168.1.210", 1
	opt.source = f"rtsp://{user}:{pwd}@{ip}//Streaming/Channels/{1}"
	# opt.source = "/home/zzd/datasets/ceyu/raw_images/9floor"
	opt.source = f"/home/zzd/datasets/quantum_element/quantum/videos/20210329134747.mp4"
	# opt.source = f"/home/zzd/datasets/quantum_element/quantum/videos/20210329141118.mp4"
	# opt.feature_save_dir = f"/home/zzd/datasets/ceyu/features_hcw3_only_skeleton/{Path(opt.source).name}"

	run(opt)


