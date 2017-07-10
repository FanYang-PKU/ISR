import sys
import gflags
import numpy as np
import cv2
from ISR import *
Flags = gflags.FLAGS
gflags.DEFINE_string('input_path', './data/Set5/woman_GT.bmp', 'input video or image path')
gflags.DEFINE_string('output_path', './data/result.png', 'output video or image path')
gflags.DEFINE_string('model_path', '../deploy-net.prototxt', 'net prototxt file')
gflags.DEFINE_string('weights_path', '../model/*.caffemodel', 'caffemodel file')
gflags.DEFINE_integer('upsample_scale', 3, 'upsampling factor')
gflags.DEFINE_boolean('image', True, 'true for image and false for video')
gflags.DEFINE_integer('frame_num', 0, 'frames to be processed')
def video_sr(input_video, output_video, upsample_scale, frame_num, model_path, weights_path):
    ISR_Net = ISR(model_path, weights_path, upsample_scale)

    videoCapture = cv2.VideoCapture(input_video)
    fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)*upsample_scale),
            int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))*upsample_scale)
    videoWriter = cv2.VideoWriter(output_video, cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), fps, size)
    success, frame = videoCapture.read()
    count=0
    while success:
        count+=1
        if count > frame_num and frame_num!=0:
            break
        hr_frame = ISR_Net.generate_hr_img(frame)
        videoWriter.write(hr_frame)
        success, frame = videoCapture.read()
    return

def readYuvFile(fp, width, height):
    uv_width = width//2
    uv_height = height//2

    Y = np.zeros((height, width), np.uint8, 'C')
    U = np.zeros((uv_height, uv_width), np.uint8,'C')
    V = np.zeros((uv_height, uv_width), np.uint8, 'C')

    Y = ord(fp.read(height*width))
    U = ord(fp.read(uv_height*uv_width))
    V = ord(fp.read(uv_height*uv_width))



    return (Y, U, V)

def writeYuvFile(fp, Y, U, V):
    fp.write(Y)
    fp.write(U)
    fp.write(V)

def image_sr(input_image, output_image, upsample_scale, model_path, weights_path):
    ISR_Net = ISR(model_path, weights_path, upsample_scale)
    im = cv2.imread(input_image)
    hr_img = ISR_Net.generate_hr_img(im)
    cv2.imwrite(output_image, hr_img)
def main(argv):
    Flags(argv)
    if Flags.image:
        image_sr(Flags.input_path, Flags.output_path, Flags.upsample_scale, Flags.model_path, Flags.weights_path)
    else:
        video_sr(Flags.input_path, Flags.output_path, Flags.upsample_scale, Flags.frame_num, Flags.model_path, Flags.weights_path)
if __name__ == '__main__':
    main(sys.argv)

