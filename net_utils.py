import numpy as np
import tensorflow as tf

import selectivesearch
import skimage


def create_graph(modelFullPath):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def get_labels(labelsFullPath):
    labfile = open(labelsFullPath, 'rb')
    lines = labfile.readlines()
    labels = [str(w.decode("utf-8")).replace("\n", "") for w in lines]
    return labels


def run_inference_on_image(sess,f,labelsFullPath):
    answer = None

    if not tf.gfile.Exists(f):
        tf.logging.fatal('File does not exist %s', f)
        return answer

    image_data3 = skimage.io.imread(f)
    image_data2 = skimage.transform.rescale(image_data3,0.25)

    img_lbl, regions = selectivesearch.selective_search(image_data2, scale=500, sigma=0.5, min_size=10)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 1000 or r['size'] > 200*200:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 2 or h / w > 2:
            continue
        candidates.add(r['rect'])

    max_score = 0
    max_answer = None
    best_region = None

    labels = get_labels(labelsFullPath)
    nb_labels = len(labels)

    for x, y, w, h in candidates:
        image_data_cropped = image_data2[y:y + h, x:x + w]
        skimage.io.imsave("/tmp/cropped_img.jpg",image_data_cropped)
        image_data = tf.gfile.FastGFile("/tmp/cropped_img.jpg", 'rb').read()

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-nb_labels:][::-1]  # Getting top 3 predictions

        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]

            if human_string=="nothing":
                image_data_cropped_ = image_data3[y * 4:(y + h) * 4, x * 4:(x + w) * 4]
                continue

            if score > max_score:
                max_answer = human_string
                max_score = score
                best_region = [x*4,y*4,(x+w)*4,(y+h)*4]
                skimage.io.imsave("/tmp/cropped_img_max.jpg", image_data_cropped)

    return max_answer,max_score,best_region