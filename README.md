# Google Object Detection API Wrapper
Simple wrapper functions for Google Object Detection API.

## Instructions
1. Create a 'detection_model_zoo' directory
2. Download model from [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md)
3. Copy the downloaded model into 'detection_model_zoo' directory
4. Check the [Jupyter notebook](https://github.com/1202kbs/object-detection-api-wrapper/blob/master/Google%20Object%20Detection%20API%20Wrapper%20Tutorial.ipynb) for further instructions and examples

## Running in Command Line
<code>python detect.py --model-name <MODEL_NAME> --url-file <URL_FILE> --downloaded <True/False></code>

*Example:* <code>python detect.py --model-name faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017 --url-file Imagenet_sample_images --downloaded True</code>

#### Required Arguments
* <code>--model-name</code> : Name of the model to use. *No Default*
* <code>--url-file</code> : Name of the url file without .json extension. *No Default*
* <code>--downloaded</code> : Indicate whether you have already downloaded model .tar file or not. *No Default*

#### Optional Arguments
* <code>--json-output-file</code> : Name of the json output file to dump results. *Default: output.json*
* <code>--n-threads</code> : Number of threads to use. *Default: 64*
* <code>--visualize</code> : Indicate whether you want to visualize the results. *Default: False*
* <code>--labels</code> : Path to the label file. *Default: object_detection/data/mscoco_label_map.pbtxt*

Run <code>python detect.py --help</code> to see a list of all options.

## Etc.
* Check the function docstring in [helpers.py](https://github.com/1202kbs/object-detection-api-wrapper/blob/master/helpers.py) for more info
* .proto files have already been compiled with Protobuf compiler(Protoc).

## Helpful Links
* Original link for [Google Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection)
* Protobuf compiler [download link](https://github.com/google/protobuf/releases/tag/v3.3.0)

## Prerequisites
* Python 3.5.x
* Tensorflow-gpu >= 1.1
* Scikit-image
* Matplotlib
* Pillow
* Pandas
* Numpy