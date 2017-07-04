# Google Object Detection API Wrapper
Simple wrapper functions for Google Object Detection API.

## Instructions

#### Method 1
1. Download the repository
2. Create a 'detection_model_zoo' directory in the extracted repository file
3. Download model from [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md)
4. Copy the downloaded model into 'detection_model_zoo' directory
5. Create a file containing the urls of images to be analyzed. This wrapper supports both [.csv](https://github.com/1202kbs/object-detection-api-wrapper/blob/master/Imagenet_sample_images.csv) and [.json](https://github.com/1202kbs/object-detection-api-wrapper/blob/master/Imagenet_sample_images.json) extensions. (Click link for example format)
6. Check the [Jupyter notebook](https://github.com/1202kbs/object-detection-api-wrapper/blob/master/Google%20Object%20Detection%20API%20Wrapper%20Tutorial.ipynb) for further instructions and examples

#### Method 2
1. Download the repository
2. Select the model to use from [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md)
3. Create a file containing the urls of images to be analyzed. This wrapper supports both [.csv](https://github.com/1202kbs/object-detection-api-wrapper/blob/master/Imagenet_sample_images.csv) and [.json](https://github.com/1202kbs/object-detection-api-wrapper/blob/master/Imagenet_sample_images.json) extensions. (Click link for example format)
4. Run detect_cm.py in command line with <code>--downloaded False</code>. Check below section for more detail

## Running in Command Line
<pre><code>python detect_cm.py --model-name MODEL_NAME --url-file URL_FILE --extension EXTENSION --downloaded DOWNLOADED --do-rescale DO_RESCALE</code></pre>

*Example:* 

<pre><code>python detect_cm.py --model-name faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017 --url-file Imagenet_sample_images --extension .json --downloaded True --do-rescale False</code></pre>

#### Required Arguments
* <code>--model-name</code> : Name of the model to use. *No Default*
* <code>--url-file</code> : Name of the url file without .json extension. *No Default*
* <code>--extension</code> : Url file file type, .json or .csv. *Default: .csv*
* <code>--downloaded</code> : Indicate whether you have already downloaded model .tar file or not. Wrapper will automatically create 'detection_model_zoo' directory and download model file if set to False. *Default: False*
* <code>--do-rescale</code> : Indicate whether you want to rescale the images or not. *Default: False*

#### Optional Arguments
* <code>--width-threshold</code> : If rescale is set to True, all the images with width size over width threshold will be rescaled to fit width threshold. *Default: 500*
* <code>--json-output-file</code> : Name of the json output file to dump results. *Default: output.json*
* <code>--n-threads</code> : Number of threads to use. *Default: 64*
* <code>--visualize</code> : Indicate whether you want to visualize the results. *Default: False*
* <code>--labels</code> : Path to the label file. *Default: object_detection/data/mscoco_label_map.pbtxt*

Run <code>python detect_cm.py --help</code> to see a list of all options.

Successful execution will create a [.json file](https://github.com/1202kbs/object-detection-api-wrapper/blob/master/output.json) which contains the results of object detection. Contents of the .json file can be read into a dictionary of numpy arrarys with json2dict function in helpers.py. For additional functions for visualization or modification of the results of object detection, look into helpers.py.

## Running within IDE
1. Import detect.py
2. Run <code>detect.object_detect(...)</code>
3. The object_detect() function takes the same arguments as command line execution. Look into detect.py for more details

If the execution is successful, the function will return a dictionary of numpy arrays containing the object detection results and create a [.json file](https://github.com/1202kbs/object-detection-api-wrapper/blob/master/output.json) which contains the results of object detection. Contents of the .json file can be read into a dictionary of numpy arrarys with json2dict function in helpers.py. For additional functions for visualization or modification of the results of object detection, look into helpers.py.

*Example:*

<pre><code>
import detect

model_name = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'
url_file = 'Imagenet_sample_images'
extension = '.csv'
downloaded = True
do_rescale = True
width_threshold = 1000
visualize = True

res = detect.object_detect(model_name=model_name, url_file=url_file, extension=extension, downloaded=downloaded, do_rescale=do_rescale, width_threshold=width_thresholdm visualize=visualize)
</code></pre>

## Etc.
* Check the function docstring in [helpers.py](https://github.com/1202kbs/object-detection-api-wrapper/blob/master/helpers.py) for more info
* .proto files have already been compiled with Protobuf compiler (Protoc)
* This code has been tested with Tensorflow 1.1 on Windows 10

## Helpful Links
* Original link for [Google Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection)
* Protobuf compiler [download link](https://github.com/google/protobuf/releases/tag/v3.3.0)

## Prerequisites
* Python 3.5.x
* Tensorflow-gpu >= 1.1
* Scikit-image
* Matplotlib
* Pandas
* Numpy