import cv2
import colorsys
import numpy as np
from matplotlib import pyplot as plt


def visual_image(imgs, titles=None, rows=1, columns=2, size=(10, 10), mode=None):
    figure = plt.figure(figsize=size)
    show_imgs = [imgs] if not isinstance(imgs, list) else imgs
    if titles is not None:
        show_titles = [titles] if not isinstance(titles, list) else titles
    else:
        if not isinstance(imgs, list):
            show_titles = ["show screen"]
        else:
            show_titles = [idx for idx in range(len(imgs))]

    for index, (img, title) in enumerate(zip(show_imgs, show_titles)):
        plt.subplot(rows, columns, index + 1)

        if not (np.min(img) > -1 and np.max(img) < 1):
            if np.any((img < 0)):
                img = np.clip(img, 0, 1)

        plt.title(title)
        if mode and mode.lower() == 'bgr2rgb':
            try:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            except:
                plt.imshow(img)
        else:
            plt.imshow(img)


def draw_boxes_on_image(image, boxes):
    temp_image = np.ones_like(image)
    for idx, box in enumerate(boxes):
        cv2.fillPoly(temp_image, [box], color=(0, 0, 255))

    image = cv2.addWeighted(image, 0.7, temp_image, 0.3, 0)
    image = np.clip(image, 0, 1)
    return image


def visual_image_with_polygons(images, polygons=None, titles=None, rows=1, columns=2, size=(10, 10), mode=None):
    images_copied = np.copy(images)
    polygons_copied = np.copy(polygons)
    figure = plt.figure(figsize=size)
    for index, (img, polygon, title) in enumerate(zip(images_copied, polygons_copied, titles)):
        plt.subplot(rows, columns, index + 1)

        if not (np.min(img) > -1 and np.max(img) < 1):
            if np.any((img < 0)):
                img = np.clip(img, 0, 1)

        plt.title(title)
        if polygon is not None:
            img = draw_boxes_on_image(img, polygon)
        if mode and mode.lower() == 'bgr2rgb':
            try:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            except:
                plt.imshow(img)
        else:
            plt.imshow(img)

    plt.show()