Usage:

1. Place the "ISR" folder into "($Caffe_Dir)/examples/"

2. To prepare the training data, we first downsample the original training images by the desired scaling factor n to form the LR images. Then we crop the LR
training images into a set of fsub*fsub pixel sub-images with a stride k. The corresponding HR sub-images (with size (n\*fsub)^2) are also cropped from the ground truth images. These LR/HR sub-image pairs are the primary training data. As we train our models with the Caffe package, its deconvolution filters will generate the output with size (n\*fsub - n + 1)^2 instead of (n\*fsub)^2. So we
also crop (n-1) pixel borders on the HR sub-images. Finally, for ×2, ×3 and ×4, we set the size of LR/HR sub-images to be 14^2/19^2, 11^2/19^2 and 10^2/21^2 with 4 pixels padding, respectively.

	Open MATLAB and direct to ($Caffe_Dir)/example/ISR/generate_training_data, open "generate\_train.m" and "generate\_test.m" and specify the following parameters:

	folder = './Train';    % folder contains training images

	savepath = '../train.h5';          % hdf5 file contains training samples

	size_input = 11;                           % There are 4 pixels padding.

	size_label = 19;                           % (11-4) *3 - 2

	scale = 3;                                 %up-sampling scale, for example 2,3,4, different scale correspond to 												different size_input and size_label, also relate to deconvolutional 											layer stride in star-sr-net.prototxt

	stride = 3;                                % crop the LR training images into a set of fsub*fsub pixel sub-images 											with a stride k

	To generate samples from other image formate, please change the  format suffixe in commond "filepaths = dir(fullfile(folder,'*.bmp'))";

	Then, run "generate\_train.m" and "generate\_test.m" to generate training and test data.

3. Put the hdf5 files path correspond to training and tesing into train.txt and test.txt

4. To train our ISR, run ./build/tools/caffe train --solver examples/ISR/solver.prototxt

5. To test our ISR, direct to evaluation, open run.sh and specify the following parameters:

    --input_path='your image or video path'

    --output_path='output path for result'

    --model_path='deploy prototxt file path'

    --weights_path='caffemodel file path'

    --upsample_scale=3

    --image=True         # set to True if you want to process image, otherwise, set to False
    
    Dependencies: OpenCV, numpy, gflags, pycaffe
    
