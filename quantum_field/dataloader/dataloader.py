import os
import torch.backends.cudnn as cudnn
from dataloader.datasets import LoadStreams, LoadImages


class Dataloader(object):
	def __init__(self, source, imgsz):
		webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')
		if webcam:
			cudnn.benchmark = True  # set True to speed up constant image size inference
			dataset = LoadStreams(source, img_size=imgsz)
			dataset.is_camera = True
		else:
			dataset = LoadImages(source, img_size=imgsz)
			dataset.is_camera = False
		self.dataset = dataset
