import os
import numpy as np
import cv2
from skimage import feature as ft
from sklearn.externals import joblib

cls_names = ["di thang", "re trai", "re phai", "dung lai", "cam coi", "qua duong", "background"]
img_label = {"straight": 0, "left": 1, "right": 2, "stop": 3, "nohonk": 4, "crosswalk": 5, "background": 6}


def preprocess_img(imgBGR, erode_dilate=True):
    """preprocess the image for contour detection.
    Args:
        imgBGR: source image.
        erode_dilate: erode and dilate or not.
    Return:
        img_bin: a binary image (blue and red).

    """
    rows, cols, _ = imgBGR.shape
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)

    Bmin = np.array([100, 43, 46])
    Bmax = np.array([124, 255, 255])
    img_Bbin = cv2.inRange(imgHSV, Bmin, Bmax)

    Rmin1 = np.array([0, 43, 46])
    Rmax1 = np.array([10, 255, 255])
    img_Rbin1 = cv2.inRange(imgHSV, Rmin1, Rmax1)

    Rmin2 = np.array([156, 43, 46])
    Rmax2 = np.array([180, 255, 255])
    img_Rbin2 = cv2.inRange(imgHSV, Rmin2, Rmax2)
    img_Rbin = np.maximum(img_Rbin1, img_Rbin2)
    img_bin = np.maximum(img_Bbin, img_Rbin)

    if erode_dilate is True:
        kernelErosion = np.ones((3, 3), np.uint8)
        kernelDilation = np.ones((3, 3), np.uint8)
        img_bin = cv2.erode(img_bin, kernelErosion, iterations=2)
        img_bin = cv2.dilate(img_bin, kernelDilation, iterations=2)

    return img_bin


def contour_detect(img_bin, min_area=0, max_area=-1, wh_ratio=2.0):
    """detect contours in a binary image.
    Args:
        img_bin: a binary image.
        min_area: the minimum area of the contours detected.
            (default: 0)
        max_area: the maximum area of the contours detected.
            (default: -1, no maximum area limitation)
        wh_ratio: the ration between the large edge and short edge.
            (default: 2.0)
    Return:
        rects: a list of rects enclosing the contours. if no contour is detected, rects=[]
    """
    rects = []
    _, contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return rects

    max_area = img_bin.shape[0] * img_bin.shape[1] if max_area < 0 else max_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area and area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            if 1.0 * w / h < wh_ratio and 1.0 * h / w < wh_ratio:
                rects.append([x, y, w, h])
    return rects


def draw_rects_on_img(img, rects):
    """ draw rects on an image.
    Args:
        img: an image where the rects are drawn on.
        rects: a list of rects.
    Return:
        img_rects: an image with rects.
    """
    img_copy = img.copy()
    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img_copy


def hog_extra_and_svm_class(proposal, clf, resize=(64, 64)):
    """classify the region proposal.
    Args:
        proposal: region proposal (numpy array).
        clf: a SVM model.
        resize: resize the region proposal
            (default: (64, 64))
    Return:
        cls_prop: propabality of all classes.
    """
    img = cv2.cvtColor(proposal, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, resize)
    bins = 9
    cell_size = (8, 8)
    cpb = (2, 2)
    norm = "L2"
    features = ft.hog(img, orientations=bins, pixels_per_cell=cell_size,
                      cells_per_block=cpb, block_norm=norm, transform_sqrt=True)
    features = np.reshape(features, (1, -1))
    cls_prop = clf.predict_proba(features)
    cls_prop = cls_prop[0]
    cls_num = clf.predict(features)
    return cls_prop


clicked = False


def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = 700
    capScale = W / width
    newX, newY = width * capScale, height * capScale
    clf = joblib.load("./svm_hog_classification/svm_model.pkl")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./output/output_webcam/webcam_output.avi', fourcc, 20.0, (width, height))
    # out_bin = cv2.VideoWriter('./output/output_webcam/webcam_output_bin.avi', fourcc, 20.0, (width, height))

    while not clicked:
        ret, img = cap.read()
        img_bin = preprocess_img(img, False)
        cv2.namedWindow("bin image", 0)
        cv2.setMouseCallback('bin image', onMouse)
        cv2.resizeWindow("bin image", int(newX), int(newY))
        # out_bin.write(img_bin)
        cv2.imshow("bin image", img_bin)
        min_area = img_bin.shape[0] * img.shape[1] / (25 * 25)
        rects = contour_detect(img_bin, min_area=min_area)
        img_bbx = img.copy()

        for rect in rects:
            xc = int(rect[0] + rect[2] / 2)
            yc = int(rect[1] + rect[3] / 2)

            size = max(rect[2], rect[3])
            x1 = max(0, int(xc - size / 2))
            y1 = max(0, int(yc - size / 2))
            x2 = min(width, int(xc + size / 2))
            y2 = min(height, int(yc + size / 2))
            proposal = img[y1:y2, x1:x2]
            cls_prop = hog_extra_and_svm_class(proposal, clf)
            cls_prop = np.round(cls_prop, 2)
            cls_num = np.argmax(cls_prop)
            cls_name = cls_names[cls_num]
            if cls_name is not "background":
                cv2.rectangle(img_bbx, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
                cv2.putText(img_bbx, cls_name, (rect[0], rect[1]), 1, 3, (0, 0, 255), 2)

        cv2.namedWindow("detect result", 0)
        cv2.setMouseCallback('detect result', onMouse)
        cv2.resizeWindow("detect result", int(newX), int(newY))
        out.write(img_bbx)
        cv2.imshow("detect result", img_bbx)
        cv2.waitKey(1)
    cap.release()
    out.release()
    # out_bin.release()
    cv2.destroyAllWindows()






