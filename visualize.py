import os
import random

import numpy as np
import time
import cv2
import json
import torch

# LSP: 1813010
# Model: RetinaNet
# Add. Loss Func.: DR loss
# Data classes: Tie, Door, Laptop

print('CUDA available: {}'.format(torch.cuda.is_available()))


def visualize(images_test_folder):

    # Draws a caption above the box in an image
    def draw_caption(image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    model_file = "target/train_model_final.pt"
    # model_file = "target/train_retinanet_4.pt"
    classes_file = "data/classes.txt"
    use_gpu = True
    threshold = 0.5
    classes = {}

    f = open(classes_file, "r")
    classes_file_info = f.read().split()
    index = 0

    while index < len(classes_file_info):
        classes[classes_file_info[index + 1].lower()] = int(classes_file_info[index])
        index += 2

    # print(classes)
    labels = {}
    for key, value in classes.items():
        labels[value] = key

    model = torch.load(model_file)

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()

    images_list = os.listdir(images_test_folder)
    images_list.remove('labels')

    random.shuffle(images_list)

    for img_name in images_list:

        image = cv2.imread(os.path.join(images_test_folder, img_name))
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()
            # print(image.shape, image_orig.shape, scale)
            scores, classification, transformed_anchors = model(image.cuda().float())
            print(f'Elapsed time: {time.time() - st}')
            idxs = np.where(scores.cpu() > threshold)
            rest_api_return = []
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label = labels[int(classification[idxs[0][j]])]
                caption = '{}'.format(label)
                draw_caption(image_orig, (x1, y1, x2, y2), caption)
                cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)
                # print([x1, y1, x2, y2, label])
                rest_api_return.append([x1, y1, x2, y2, label])

            cv2.imshow('detections', image_orig)
            cv2.waitKey(0)
    return json.dumps(str(rest_api_return))

if __name__ == '__main__':
    labels_test_folder = "data/multidata/test/labels/"
    images_test_folder = "data/multidata/test/"
    visualize(images_test_folder)