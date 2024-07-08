import os
import random
import shutil


class LabelUtils:

    def create_shuffled_images_folder(self):
        if not os.path.exists("shuffled_images"):
            os.mkdir("shuffled_images")

        image_files = [f for f in os.listdir("images") if f.endswith(".jpg")]
        random.shuffle(image_files)

        for img in image_files:
            os.rename(
                f"images/{img}", f"shuffled_images/img_{len(os.listdir('shuffled_images'))}.jpg")

    def create_labeled_images_zip_file(self):
        if not os.path.exists("obj"):
            os.mkdir("obj")

        file_prefixes = [f.split('.')[0] for f in os.listdir(
            "shuffled_images") if f.endswith(".txt")]

        for prefix in file_prefixes:
            os.rename(f"shuffled_images/{prefix}.txt", f"obj/{prefix}.txt")
            os.rename(f"shuffled_images/{prefix}.jpg", f"obj/{prefix}.jpg")

        shutil.make_archive('yolov4-tiny/obj', 'zip', '.', 'obj')

    def update_config_files(self, classes):
        # Overwriting the existing obj.names file with the classes (i.e. objects such as player, enemies, items) that will be detected which will be provided
        with open("./yolov4-tiny/obj.names", "w") as file:
            file.write("\n".join(classes))

        # Open/read the contents of the template config for training the model
        with open("./yolov4-tiny/yolov4-tiny-custom_template.cfg", 'r') as file:
            cfg_content = file.read()

        # Modify the config file and save the newly modified contents of the config file as variable "updated_cfg_content"
        updated_cfg_content = cfg_content.replace(
            '_CLASS_NUMBER_', str(len(classes)))
        updated_cfg_content = updated_cfg_content.replace(
            '_NUMBER_OF_FILTERS_', str((len(classes) + 5) * 3))
        updated_cfg_content = updated_cfg_content.replace(
            '_MAX_BATCHES_', str(max(6000, len(classes) * 2000)))

        # Overwrite the existing (or create new if doesn't exist) file which will be your main config file for training the model
        with open("./yolov4-tiny/yolov4-tiny-custom.cfg", 'w') as file:
            file.write(updated_cfg_content)
