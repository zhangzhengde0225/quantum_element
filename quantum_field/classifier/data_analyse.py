import os
import shutil
from pathlib import Path
import numpy as np
import random


def analyse():
	feature_dir = '/home/zzd/datasets/ceyu/features_hcw3_only_skeleton'
	feature_dir = '/home/zzd/datasets/ceyu/features_hcw3_only_skeleton_trteval/val'
	# out_dir = '/home/zzd/datasets/ceyu/features_hcw3_only_skeleton_trteval'
	sub_folders = [x for x in os.listdir(feature_dir) if os.path.isdir(f'{feature_dir}/{x}')]

	# sub_folders = [x for x in sub_folders if x.startswith('fighting')]
	print(sub_folders)
	imgs = []
	for i, s in enumerate(sub_folders):
		file = [f'{s}/{x}' for x in os.listdir(f'{feature_dir}/{s}')]
		imgs.extend(file)
		print(len(file), file[:2])
	print(len(imgs), imgs[:2])
	classes = [str(Path(x).stem).split('_')[1] for x in imgs]
	print(len(classes), classes[:2])
	classes = np.unique(classes)
	print(len(classes), classes)
	imgs_in_cls = []
	num = 1200  # 小于1200的全部取，大于1200的随机取1200
	for i, cls in enumerate(classes):
		img = [x for x in imgs if str(Path(x).stem).split('_')[1] == cls]
		print(f'{i} {cls:<10} {len(img):<5} ', end='')
		img = random.sample(img, num) if len(img) > num else img
		print(f'{len(img)}')
		imgs_in_cls.append(img)

	exit()
	print(len(imgs_in_cls))
	ratio_trteval = np.array([8, 2, 2])  # 比例8:2:2，800张，200张，200张
	trtrval = ['train', 'test', 'val']
	for i, img in enumerate(imgs_in_cls):
		r = ratio_trteval / np.sum(ratio_trteval)
		num_sample = np.array(r * len(img), dtype=np.int32)
		# print(r, num_sample, len(img))
		np.random.shuffle(img)
		for j, trte in enumerate(trtrval):
			start = np.sum(num_sample[:j])
			end = np.sum(num_sample[:j+1])
			trtr_img = img[start:end]
			# print(i, j, start, end, len(trtr_img))

			for k, p in enumerate(trtr_img):
				sp = f'{feature_dir}/{p}'
				tp = f'{out_dir}/{trtrval[j]}/{p}'
				if not os.path.exists(Path(tp).parent):
					os.makedirs(Path(tp).parent)
				shutil.copy(sp, tp)
				print(f'\rcopy: [{i+1}/{len(imgs_in_cls)}] {trtrval[j]:<5} [{k+1:>3}/{len(trtr_img)}] {tp}', end='')
				# print(sp, tp)
				# exit()
	print('finished')


def anaylise_label():
	rp = "/home/zzd/datasets/ceyu/labels"
	folds = ['fighting_kitchen', 'fighting_street', '9floor', '10floor', '11floor']
	ret = []
	for i, fold in enumerate(folds):
		p = f'{rp}/{fold}'
		files = [f'{p}/{x}' for x in os.listdir(p) if x.endswith('.txt')]
		valid_files = []
		for file in files:
			with open(file, 'r') as f:
				data = f.readlines()
			has_label = False
			for line in data:
				if 'status:' in line:
					if line.split('status:')[-1].split()[-1] == 'unknown':
						continue
					else:
						has_label = True
						# print(line.split()[-2:])
					break
			valid_files.append(file) if has_label else None
			print(f'\r{p} {len(valid_files)} {has_label}', end='')
		print('')
		ret.append([Path(p).stem, valid_files])

	return ret


if __name__ == '__main__':
	analyse()
	# anaylise_label()