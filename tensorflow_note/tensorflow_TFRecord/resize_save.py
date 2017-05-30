from PIL import Image
import os


def resize_save(path, image_W, image_H):
    for image_name in os.listdir(path):
        image_new_path = path + image_name
        img = Image.open(image_new_path)

        new_img = img.resize((image_W, image_H))     
        new_img.save("new_img_"+image_name+".jpg")

image_raw_path = './dog/'
resize_save(image_raw_path, 64, 64)