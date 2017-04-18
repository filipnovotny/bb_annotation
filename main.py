"""
Preparing model:
 - Install bazel ( check tensorflow's github for more info )
    Ubuntu 14.04:
        - Requirements:
            sudo add-apt-repository ppa:webupd8team/java
            sudo apt-get update
            sudo apt-get install oracle-java8-installer
        - Download bazel, ( https://github.com/bazelbuild/bazel/releases )
          tested on: https://github.com/bazelbuild/bazel/releases/download/0.2.0/bazel-0.2.0-jdk7-installer-linux-x86_64.sh
        - chmod +x PATH_TO_INSTALL.SH
        - ./PATH_TO_INSTALL.SH --user
        - Place bazel onto path ( exact path to store shown in the output)
- For retraining, prepare folder structure as
    - root_folder_name
        - class 1
            - file1
            - file2
        - class 2
            - file1
            - file2
- Clone tensorflow
- Go to root of tensorflow
- bazel build tensorflow/examples/image_retraining:retrain
- bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir /path/to/root_folder_name  --output_graph /path/output_graph.pb --output_labels /path/output_labels.txt --bottleneck_dir /path/bottleneck
** Training done. **
For testing through bazel,
    bazel build tensorflow/examples/label_image:label_image && \
    bazel-bin/tensorflow/examples/label_image/label_image \
    --graph=/path/output_graph.pb --labels=/path/output_labels.txt \
    --output_layer=final_result \
    --image=/path/to/test/image
For testing through python, change and run this code.
"""

import numpy as np
import tensorflow as tf
import time

from PIL import Image, ImageDraw, ImageFont
import selectivesearch
import skimage
import shutil

imagePath = '/home/dt/Videos/notilo/tulips/frames/tulips_{0}.jpg'
imageCopyPath = '/home/dt/Videos/notilo/tulips_detected/tulips_{0}.jpg'
modelFullPath = '/tmp/output_graph.pb'
labelsFullPath = '/tmp/output_labels.txt'


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(sess,num):
    answer = None

    if not tf.gfile.Exists(imagePath.format(num)):
        tf.logging.fatal('File does not exist %s', imagePath.format(num))
        return answer

    image_data3 = skimage.io.imread(imagePath.format(num))
    image_data2 = skimage.transform.rescale(image_data3,0.25)

    img_lbl, regions = selectivesearch.selective_search(
        image_data2, scale=500, sigma=0.5, min_size=10)

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
    print("len(candidates)=",len(candidates))
    for x, y, w, h in candidates:
        image_data_cropped = image_data2[y:y + h, x:x + w]
        skimage.io.imsave("/tmp/cropped_img.jpg",image_data_cropped)
        image_data = tf.gfile.FastGFile("/tmp/cropped_img.jpg", 'rb').read()
        # image_data = tf.gfile.FastGFile(imagePath.format(num), 'rb').read()

        #image_data = tf.image.crop_to_bounding_box(image_data, offset_height, offset_width, target_height, target_width)

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)
        top_k = predictions.argsort()[-3:][::-1]  # Getting top 3 predictions
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w.decode("utf-8")).replace("\n", "") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]

            if human_string=="nothing":
                image_data_cropped_ = image_data3[y * 4:(y + h) * 4, x * 4:(x + w) * 4]
                skimage.io.imsave("/home/dt/Videos/notilo/nothing/nothing{0}.jpg".format(int(time.time())),image_data_cropped_)
                continue

            if score > max_score:
                max_answer = human_string
                max_score = score
                best_region = [x*4,y*4,(x+w)*4,(y+h)*4]
                skimage.io.imsave("/tmp/cropped_img_max.jpg", image_data_cropped)

    print('%s (score = %.5f)' % (max_answer, max_score))
    return max_answer,max_score,best_region


if __name__ == '__main__':
    # Creates graph from saved GraphDef.
    create_graph()
    last_start=0
    max_dist=0
    cur_dist=0
    with tf.Session() as sess:
        for i in range(515,2028):
            print("image number {0}".format(i))

            answer,score,best_region  =run_inference_on_image(sess,i)
            # get an image
            base = Image.open(imagePath.format(i)).convert('RGBA')
            print(answer,score)

            if (answer=="tulips" ) and score>0.8:
                if(cur_dist>max_dist):
                    last_start = i
                    max_dist = cur_dist

                cur_dist+=1
                # make a blank image for the text, initialized to transparent text color
                txt = Image.new('RGBA', base.size, (255, 255, 255, 0))

                # get a font
                fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
                # get a drawing context
                d = ImageDraw.Draw(txt)

                # draw text, half opacity
                # d.text((10, 10), answer, font=fnt, fill=(255, 255, 255, 128))
                # draw text, full opacity
                d.text((10, 60), answer, font=fnt, fill=(255, 255, 255, 255))
                d.rectangle(best_region,outline='red')

                out = Image.alpha_composite(base, txt)
                out.save(imageCopyPath.format(i))
                shutil.copy("/tmp/cropped_img_max.jpg","/home/dt/Videos/notilo/tulip_candidates/tulips_{0}.jpg".format(int(time.time())))
                del out
                del txt
            elif answer=="nothing" and score>0.8:
                base.save("/home/dt/Videos/notilo/dandelion_candidates/tulips_{0}.jpg".format(int(time.time())))
            else:
                cur_dist = 0
            #     base.save(imageCopyPath.format(i))

            del base

        print(last_start,max_dist)