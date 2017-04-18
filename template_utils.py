from jinja2 import Environment, FileSystemLoader, select_autoescape
import shutil
import os


class PascalDataset(object):
    def __init__(self,base_dir,dataset_dir,labels):
        self.env = Environment(
            loader=FileSystemLoader('%s/templates/' % os.path.dirname(__file__)),
            autoescape=select_autoescape(['html', 'xml'])
        )

        self.labels = labels

        self.bdir = os.path.join(base_dir, dataset_dir)
        if os.path.exists(self.bdir):
            shutil.rmtree(self.bdir)

        self.annotations = os.path.join(self.bdir,"Annotations")
        self.image_sets = os.path.join(self.bdir, "ImageSets")
        self.image_sets_main = os.path.join(self.image_sets, "Main")
        self.jpeg_images = os.path.join(self.bdir, "JPEGImages")

        os.makedirs(self.bdir)
        os.makedirs(self.annotations)
        os.makedirs(self.image_sets)
        os.makedirs(self.image_sets_main)
        os.makedirs(self.jpeg_images)

        self.jpeg_images_file = os.path.join(self.jpeg_images, "{0:06d}.jpg")
        self.annotations_file = os.path.join(self.annotations, "{0:06d}.xml")
        self.image_sets_main_file = os.path.join(self.image_sets_main, "{0}_train.txt")
        self.image_sets_main_trainfile = os.path.join(self.image_sets_main, "train.txt")

        self.folder_name = os.path.basename(dataset_dir)


    def write_image_with_label_and_bbox(self,path,image,number,label,bbox):
        template = self.env.get_template('annotation.tpl.xml')

        annotation_text = template.render(
                                folder_name = self.folder_name,
                                file_number = "{0:06d}".format(number),
                                img = image,
                                bbox = bbox,
                                label = label
                            )
        with open(self.annotations_file.format(number), "w") as annotation_file:
            annotation_file.write(annotation_text)



        shutil.copy(path,self.jpeg_images_file.format(number))

        for l in self.labels:
            with open(self.image_sets_main_file.format(l), "a+") as specific_train_file:
                if l == label:
                    specific_train_file.write("{0:06d}  1\n".format(number))
                else:
                    specific_train_file.write("{0:06d} -1\n".format(number))

        with open(self.image_sets_main_trainfile, "a+") as train_file:
            train_file.write("{0:06d}\n".format(number))