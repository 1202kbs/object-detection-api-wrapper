import tarfile
import urllib
import codecs
import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from skimage.transform import rescale
from skimage import io

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util


DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
IMAGE_SIZE = (12, 8)


def load_category_index(path_to_labels, num_classes=90):
    """
    Load category index for classes.

    :param path_to_labels: (String) Path to category index.
    :param num_classes: (Integer) Number of classes. 90 by default.
    :return: (Dictionary) Dictionary of category index of format {class_id: {'id': class_id, 'name': class_name}}.
    """

    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return category_index


def download_extract_model(model_file, downloaded=True, download_base=DOWNLOAD_BASE):
    """
    Optionally downloads the model file and extracts the frozen model into project 'detection_model_zoo' directory.

    :param model_file: (String) Name of the model file intending to use.
    :param downloaded: (Boolean) If set to False, function will download and extract the .tar file. If set to True, the
     function will look into 'detection_model_zoo' directory for the .tar file.
    :param download_base: (String) The download base for Google Object Recognition API Tensorflow Detection Model Zoo.
     http://download.tensorflow.org/models/object_detection/ by default.
    :return: None
    """

    if not os.path.exists('detection_model_zoo'):
    	os.mkdir('detection_model_zoo')

    if not downloaded:
        print('Downloading Model: ' + model_file)
        opener = urllib.request.URLopener()
        opener.retrieve(download_base + model_file, 'detection_model_zoo/' + model_file)
        print('Download Complete')
    
    print('Extracting Model: ' + model_file)
    tar_file = tarfile.open('detection_model_zoo/' + model_file)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())
    print('Extraction Complete')


def json2csv(json_name, csv_name):
    """
    Converts json file of urls into a csv file with headers 'img_id' and 'url', respectively.

    :param json_name: (String) Name of the json file containing image urls.
    :param csv_name: (String) Name of the csv file to be produced.
    :return: None
    """

    data = pd.read_json(json_name)
    data.columns = ['img_id', 'img_url']
    data.set_index('img_id', inplace=True)
    data.to_csv(csv_name, header=True)


def url_image_reader(urls, do_rescale=False, width_threshold=500):
    """
    Reads the list of image urls and converts them to tensors.

    :param urls: (List) List of image urls
    :param do_rescale: (Boolean) Indicate whether you wish to rescale the image or not
    :param width_threshold: (Integer) If do_rescale=True, all the images with width size over width_threshold will be rescaled to fit width_threshold
    :return: (Tensor) Tensor of images specified by the list of url
    """

    def rescale_img(img):
    	if np.shape(img)[1] <= 500:
    		return img
    	else:
    		scaling_factor = width_threshold / np.shape(img)[1]
    		return rescale(img, scaling_factor)

    def load_nth_url(n):
    	if do_rescale:
    		return rescale_img(io.imread(urls[n])).astype(np.uint8)
    	else:
    		return io.imread(urls[n])
    
    print('Reading URLs...')
    url_iter = tf.Variable(0, name='url_iter').count_up_to(len(urls))
    return tf.py_func(load_nth_url, [url_iter], stateful=True, Tout=[tf.uint8])


def threshold_accuracy(threshold, res):
    """
    Removes any object detection results whose confidence (score) is below certain point specified by the user.

    :param threshold: (Float) All the results whose confidence is below the threshold will be removed. Must be a
     nonnegative float smaller than 1.
    :param res: (Dictionary) The dictionary of numpy arrays returned by detect.py function.
    :return: (Dictionary) The dictionary of numpy arrays res with results below the threshold removed.
    """

    ret = {}
    for key, value in res.items():
        idx = value['scores'] > threshold
        ret[key] = {'boxes': value['boxes'][idx],
                    'scores': value['scores'][idx],
                    'classes': value['classes'][idx],
                    'num_detections': [np.sum(idx)]}

    return ret


def class2str(category_index, res):
    """
    Converts the numpy array of integers which indicate the classes of objects detected into a list of String that
     explicitly indicate the names of objects detected.

    :param category_index: (Dictionary) The dictionary of indices and classes returned by load_category_index function.
    :param res: (Dictionary) The dictionary of numpy arrays returned by detect.py function.
    :return: (Dictionary) Content identical to res, except that classes list now contain Strings of object names.
    """

    ret = {}
    for key, value in res.items():
        classes2str = [category_index[i]['name'] for i in value['classes']]
        ret[key] = {'boxes': value['boxes'],
                    'scores': value['scores'],
                    'classes': classes2str,
                    'num_detections': value['num_detections']}

    return ret


def visualize(csv_name, res, category_index, image_size=IMAGE_SIZE):
    """
    Visualizes the object detection results using matplotlib.

    :param csv_name: (String) Name of the csv file that contains the list of image urls (returned by json2csv function).
    :param res: (Dictionary) The dictionary of numpy arrays returned by detect.py function.
    :param category_index: (Dictionary) The dictionary of indices and classes returned by load_category_index function.
    :param image_size: (Tuple) Tuple of integers indicating the size of image to be visualized. (12, 8) by default.
    :return: None
    """

    li = pd.read_csv(csv_name)
    li.set_index('img_id', inplace=True)
    data = li.to_dict(orient='index')

    for key, value in res.items():
        url = data[key]['img_url']
        image_np = io.imread(url)

        vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                           res[key]['boxes'],
                                                           res[key]['classes'],
                                                           res[key]['scores'],
                                                           category_index,
                                                           use_normalized_coordinates=True,
                                                           line_thickness=8)
        plt.figure(figsize=image_size)
        plt.imshow(image_np)
        plt.show()


def dict2json(res, json_name):
    """
    Dumps the results of object detection into a json file.

    :param res: (Dictionary) The dictionary of numpy arrays returned by detect.py function.
    :param json_name: (String) Name of the json file to be produced.
    :return: None
    """

    ret = {}

    for key, value in res.items():
        ret[key] = {'boxes': res[key]['boxes'].tolist(),
                    'scores': res[key]['scores'].tolist(),
                    'classes': res[key]['classes'].tolist(),
                    'num_detections': res[key]['num_detections'].tolist()}

    json.dump(ret, codecs.open(json_name, 'w', encoding='utf-8'))


def json2dict(json_name):
    """
    Reads the json file containing the results of object detection and returns a dictionary of numpy arrays.

    :param json_name: (String) Name of the json file containing object detection results.
    :return: (Dictionary) The dictionary of numpy arrays returned by the detect.py function.
    """

    obj_text = codecs.open(json_name, 'r', encoding='utf-8').read()
    res = json.loads(obj_text)

    ret = {}

    for key, value in res.items():
        ret[key] = {'boxes': np.array(res[key]['boxes']),
                    'scores': np.array(res[key]['scores']),
                    'classes': np.array(res[key]['classes']),
                    'num_detections': np.array(res[key]['num_detections'])}

    return ret
