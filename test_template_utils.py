from template_utils import  PascalDataset
import skimage.io

path = "/home/dt/res/flower_photos/tulips/112334842_3ecf7585dd.jpg"
img = skimage.io.imread(path)

p = PascalDataset("/home/dt/res","voc",["dandelion","tulips","nothing"])
p.write_image_with_label_and_bbox(path,img,12,"tulips",(2,5,96,36))
p.write_image_with_label_and_bbox(path,img,13,"tulips",(2,500,96,36))
p.write_image_with_label_and_bbox(path,img,14,"dandelion",(2,5,96,36))