import cv2
from alphapose.utils.bbox import _box_to_center_scale, _center_scale_to_box
from alphapose.utils.transforms import get_affine_transform, im_to_torch


def test_transform(src, bbox, input_size):
	xmin, ymin, xmax, ymax = bbox
	center, scale = _box_to_center_scale(
		xmin, ymin, xmax - xmin, ymax - ymin, float(input_size[1]/input_size[0]))
	scale = scale * 1.0

	inp_h, inp_w = input_size

	trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
	img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
	bbox = _center_scale_to_box(center, scale)

	img = im_to_torch(img)
	img[0].add_(-0.406)
	img[1].add_(-0.457)
	img[2].add_(-0.480)

	return img, bbox