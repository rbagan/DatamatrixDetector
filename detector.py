import cv2
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--image', help='')
parser.add_argument('--blockSize', type=int, help='Size of the neighbourhood', default=4)
parser.add_argument('--kSize', type=int, help='Sobel kernel size', default=3)
parser.add_argument('--k', type=float, help='Harris free parameter', default=0.06)
parser.add_argument('--thr', type=float, help='Threshold', default=0.001)
parser.add_argument('--thrArea', type=float, help='Box area threshold', default=0.001)
parser.add_argument('--matches', type=int, help='Box area threshold', default=100)
args = parser.parse_args()

os.makedirs('output', exist_ok=True)

gt = cv2.imread('gt.png')
gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
gt = cv2.resize(gt, (800, 600), interpolation=cv2.INTER_CUBIC)

with open('output_coords.txt', 'w') as coords_file:
    img = cv2.imread(args.image)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    harris = cv2.cornerHarris(img_gray, args.blockSize, args.kSize, args.k)

    _, thr = cv2.threshold(harris, args.thr * harris.max(), 255, cv2.THRESH_BINARY)
    thr = thr.astype('uint8')

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = list(filter(lambda x: cv2.contourArea(cv2.convexHull(x)) > args.thrArea, contours))

    for i, contour in enumerate(contours):
        mask = np.zeros_like(img)
        cv2.drawContours(mask, contour, -1, (255, 255, 255), -1)
        cv2.fillPoly(mask, pts=[contour], color=(255, 255, 255))

        out = np.zeros_like(img)
        out[mask == 255] = img[mask == 255]

        coords = np.argwhere(mask == 255)
        topx = np.inf
        topy = np.inf
        bottomx = 0
        bottomy = 0

        for x, y, _ in coords:
            topx = min(x, topx)
            topy = min(y, topy)
            bottomx = max(x, bottomx)
            bottomy = max(y, bottomy)

        crop = img[topx:bottomx+1, topy:bottomy+1]
        crop_filename = f'output/{args.image}_crop_{i}.png'

        sift = cv2.SIFT_create()

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (800,600), interpolation=cv2.INTER_CUBIC)

        kp1, des1 = sift.detectAndCompute(gray,None)
        kp2, des2 = sift.detectAndCompute(gt,None)

        bf = cv2.BFMatcher()

        matches = bf.match(des1,des2)

        if len(matches) > args.matches:
            cv2.imwrite(crop_filename, crop)
            coords_file.write(f'{topx},{topy},{bottomx},{bottomy}\n')
