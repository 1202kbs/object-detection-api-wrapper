from argparse import ArgumentParser
import time
import os

import tensorflow as tf
import pandas as pd
import numpy as np

import helpers


EXTENSION = '.csv'
DOWNLOADED = False
JSON_OUTPUT_FILE = 'output.json'
N_THREADS = 64
VISUALIZE = False
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')
BATCH_SIZE = 1
DO_RESCALE = True
WIDTH_THRESHOLD = 500

def object_detect(model_name, url_file, extension=EXTENSION, downloaded=DOWNLOADED, do_rescale=DO_RESCALE, width_threshold=WIDTH_THRESHOLD, json_output_file=JSON_OUTPUT_FILE, n_threads=N_THREADS, visualize=VISUALIZE, path_to_labels=PATH_TO_LABELS):
   	
    model_file = model_name + '.tar.gz'
    path_to_ckpt = model_name + '/frozen_inference_graph.pb'
    
    json_url_file = url_file + '.json'
    csv_url_file = url_file + '.csv'

    helpers.download_extract_model(model_file=model_file, downloaded=downloaded)
    
    if extension == '.json':    
        helpers.json2csv(json_name=json_url_file, csv_name=csv_url_file)

    li = pd.read_csv(csv_url_file)
    url_li = li['img_url'].tolist()
    id_li = li['img_id'].tolist()

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        im = helpers.url_image_reader(url_li, do_rescale=do_rescale, width_threshold=width_threshold)

        queue = tf.PaddingFIFOQueue(capacity=500, dtypes=tf.uint8, shapes=[(None, None, None)])
        enq_op = queue.enqueue(im)
        inputs = queue.dequeue_many(BATCH_SIZE)
        qr = tf.train.QueueRunner(queue, [enq_op] * n_threads)

        with tf.Session(graph=detection_graph) as sess:

            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            enqueue_threads = qr.create_threads(sess=sess, coord=coord, start=True)

            conversion_time = []
            ix = 1
            res = {}

            category_index = helpers.load_category_index(path_to_labels)

            t = time.time()

            try:
                while not coord.should_stop():
                    image = sess.run(inputs)  # Tensor of dimension (1, None, None, 3)

                    print('Processing Image', ix)

                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                    t2 = time.time()

                    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                                        feed_dict={image_tensor: image})

                    conversion_time.append(time.time() - t2)

                    print('Image', ix, 'Processing Time:', conversion_time[ix - 1], 'sec')

                    res[id_li[ix - 1]] = {'boxes': np.squeeze(boxes),
                                          'scores': np.squeeze(scores),
                                          'classes': np.squeeze(classes).astype(np.int32),
                                          'num_detections': num_detections}

                    ix += 1

            except tf.errors.OutOfRangeError:
                print('Total Image Processing Time:', sum(conversion_time), 'sec')
                print('Total Time Consumed:', time.time() - t, 'sec')

            finally:
                coord.request_stop()

                helpers.dict2json(res, json_output_file)

                if visualize:
                    helpers.visualize(csv_url_file, res, category_index)

            coord.join(enqueue_threads)

    return res
