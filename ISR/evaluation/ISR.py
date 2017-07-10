import numpy as np
import cv2
import sys
caffe_root = '../../../python'
sys.path.insert(0, caffe_root)
import caffe
class ISR(object):
    def __init__(self, model_path, weight_path, upsample_scale):
        caffe.set_mode_gpu()
        self.net = caffe.Net(model_path, weight_path, caffe.TEST)
        self.upsample_scale = upsample_scale

    def generate_hr_img(self, lr_img):
        if lr_img.shape[2] == 3:
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2YCR_CB)

        hr_img = cv2.resize(lr_img, None, fx=self.upsample_scale, fy=self.upsample_scale, interpolation=cv2.INTER_CUBIC)
        caffe_input = lr_img[:, :, 0]
        caffe_input = caffe_input.astype(np.float32)
        caffe_input = caffe_input/255
        self.net.blobs['data'].reshape(1, 1, caffe_input.shape[0], caffe_input.shape[1])
        self.net.blobs['data'].data[0] = caffe_input
        self.net.forward()
        caffe_output = self.net.blobs['conv3'].data[...]*255
        caffe_output = np.squeeze(caffe_output)
        np.clip(caffe_output, 0, 255, out=caffe_output)
        if self.upsample_scale == 2:
            hr_img[0:-1, 0:-1, 0] = caffe_output.astype(np.uint8)
        elif self.upsample_scale == 3:
            hr_img[1:-1, 1:-1, 0] = caffe_output.astype(np.uint8)
        if lr_img.shape[2] == 3:
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_YCR_CB2BGR)

        return hr_img






