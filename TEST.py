import numpy as np
import cv2


def test1():
    blank_img = np.zeros((3000, 2000, 3)).astype(np.uint8)
    blank_img = cv2.resize(blank_img, (1500, 2000))
    print('test1')

def test2():
    blank_img2 = np.zeros((3000, 2000, 3)).astype(np.uint8)
    blank_img2 = cv2.resize(blank_img2, (1500, 2000))
    print('test2')

test1()
test2()