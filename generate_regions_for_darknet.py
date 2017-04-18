import tensorflow as tf
import time

from PIL import Image, ImageDraw, ImageFont
import skimage.io
import shutil
from net_utils import run_inference_on_image,create_graph,get_labels

from template_utils import PascalDataset

imagePath = '/home/dt/Videos/notilo/tulips/frames/tulips_{0}.jpg'
imageCopyPath = '/home/dt/Videos/notilo/tulips_detected/tulips_{0}.jpg'
modelFullPath = '/home/dt/res/inceptionv3_trained_on_flowers/flower_graph_informed_with_nothing_gen4.pb'
labelsFullPath = '/tmp/output_labels.txt'


if __name__ == '__main__':
    create_graph(modelFullPath)
    last_start=0
    max_dist=0
    cur_dist=0
    labels  =get_labels(labelsFullPath)
    p = PascalDataset("/home/dt/res", "voc", labels)
    with tf.Session() as sess:
        for i in range(515,2028):
            path = imagePath.format(i)
            print("image number {0}".format(i))

            answer,score,best_region  =run_inference_on_image(sess,path,labelsFullPath)
            img = skimage.io.imread(path)
            p.write_image_with_label_and_bbox(path,img,i-515,answer,best_region)
            # get an image
            print(answer,score)
