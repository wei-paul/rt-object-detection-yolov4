import cv2 as cv
import numpy as np
from time import time
from windowcapture import WindowCapture
from labeldataset import LabelUtils
from imageprocessor import ImageProcessor

# WindowCapture.list_window_names()
# wincap = WindowCapture('Vampire Survivors')
# wincap.generate_image_dataset()

# # If you're not going to label all the generated images, make sure to shuffle them. This it to ensure that you will cover a wide range of scenarios
# # This will also avoid any bias towards specific patterns/sequences
# # The function below shuffles the images in the images folder and inserts them into the shuffled_images folder
# lbUtils = LabelUtils()
# lbUtils.create_shuffled_images_folder()

# # After labeling the images, extract the content of the zip file from makesense.ai
# # Copy .txt yolo label files to "shuffled_images" folder
# # Run below function to generate zip file with images/labels inside yolov4-tiny folder
# lbUtils.create_labeled_images_zip_file()

# # Fill this list with the classes (labels) used
# # *Make sure to enter exact same classes and in the exact same order* otherwise model won't work

# classes = ["enemies"]

# lbUtils.update_config_files(classes)

# loop_time = time()

# while (True):

#     screenshot = wincap.get_screenshot()

#     cv.imshow('Computer Vision', screenshot)

#     # print('FPS {}'.format(1 / (time()-loop_time)))
#     loop_time = time()
#     # press 'q' with the output window focused to exit.
#     # waits 1 ms every loop to process key presses
#     if cv.waitKey(1) == ord('q'):
#         cv.destroyAllWindows()
#         break

# print('Done')

window_name = "Vampire Survivors"
cfg_file_name = "./yolov4-tiny/yolov4-tiny-custom.cfg"
weights_file_name = "yolov4-tiny-custom_last.weights"

wincap = WindowCapture(window_name)
improc = ImageProcessor(wincap.get_window_size(),
                        cfg_file_name, weights_file_name)


while (True):

    ss = wincap.get_screenshot()

    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

    coordinates = improc.process_image(ss)

    # for coordinate in coordinates:
    #     print(coordinate)

    # If you have limited computer resources, consider adding a sleep delay between detections.
    # sleep(0.2)

print('Finished.')
