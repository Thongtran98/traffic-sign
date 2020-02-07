import os
import numpy as np 
import cv2
from skimage import feature as ft 
from sklearn.externals import joblib

cls_names = ["di thang", "re trai", "re phai", "dung lai", "cam coi", "qua duong", "background"]
img_label = {"straight": 0, "left": 1, "right": 2, "stop": 3, "nohonk": 4, "crosswalk": 5, "background": 6}

def preprocess_img(imgBGR, erode_dilate=True):
    """preprocess the image for contour detection.  --Tiền xử lý ảnh trước khi phát hiện đường viền.
    Args:
        imgBGR: source image.   --Ảnh gốc trước khi xử lý.
        erode_dilate: erode and dilate or not.  --co, dãn ảnh.
    Return:
        img_bin: a binary image (blue and red). --Ảnh nhị phân.

    """
    rows, cols, _ = imgBGR.shape    #hàng cột và ngưỡng màu của ảnh

    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)    #chuyển ảnh màu sang ảnh HSV (H = Hue: màu sắc,
                                                        # S = Saturation: độ đậm đặc, sự bảo hòa, V = value: giá trị cường độ sáng)

    Bmin = np.array([100, 43, 46])
    Bmax = np.array([124, 255, 255])
    img_Bbin = cv2.inRange(imgHSV,Bmin, Bmax)   #xác định khoảng màu cầu tìm nhờ HSV
    
    Rmin1 = np.array([0, 43, 46])
    Rmax1 = np.array([10, 255, 255])
    img_Rbin1 = cv2.inRange(imgHSV,Rmin1, Rmax1)
    
    Rmin2 = np.array([156, 43, 46])
    Rmax2 = np.array([180, 255, 255])
    img_Rbin2 = cv2.inRange(imgHSV,Rmin2, Rmax2)
    img_Rbin = np.maximum(img_Rbin1, img_Rbin2)
    img_bin = np.maximum(img_Bbin, img_Rbin)

    if erode_dilate is True:
        """tạo ra hàm toàn số 1"""
        kernelErosion = np.ones((9,9), np.uint8)
        kernelDilation = np.ones((9,9), np.uint8)
        img_bin = cv2.erode(img_bin, kernelErosion, iterations=2)   #co hình ảnh số lần lặp lại 2
        img_bin = cv2.dilate(img_bin, kernelDilation, iterations=2) #giãn hình ảnh số lần lặp lại 2 chống nhiễu

    return img_bin


def contour_detect(img_bin, min_area=0, max_area=-1, wh_ratio=2.0):
    """detect contours in a binary image.   --Phát hiện đường viền trong ảnh nhị phân.
    Args:
        img_bin: a binary image.    --Ảnh nhị phân.
        min_area: the minimum area of the contours detected.    --diện tích tối thiểu của các đường viền được  phát hiên.
            (default: 0)
        max_area: the maximum area of the contours detected.    --diện tích tối đa của các đường viền được  phát hiên.
            (default: -1, no maximum area limitation)
        wh_ratio: the ration between the large edge and short edge.     --tỷ lệ cạnh dài và cạnh ngắn
            (default: 2.0)
    Return:
        rects: a list of rects enclosing the contours. if no contour is detected, rects=[]
        --danh sách các hình chữ nhật bao quanh các đường viền.Nếu không có đường viền rects=[]
    """
    """
    """
    rects = []
    _, contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #tìm đường tất cả đường viền trong ảnh
    if len(contours) == 0:
        return rects

    max_area = img_bin.shape[0]*img_bin.shape[1] if max_area<0 else max_area    #chiều_cao_ảnh_img_bin * chiều_dài_img_bin
    for contour in contours:
        area = cv2.contourArea(contour) #tính toán khu vực đường biên
        if area >= min_area and area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)  #Hàm tính toán và trả về hình chữ nhật giới hạn tối thiểu cho tập hợp điểm đã chỉ định
            if 1.0*w/h < wh_ratio and 1.0*h/w < wh_ratio:
                rects.append([x,y,w,h])
    return rects


def draw_rects_on_img(img, rects):
    """ draw rects on an image.     --Vẽ hình chữ nhật trên ảnh.
    Args:
        img: an image where the rects are drawn on.     --Ảnh đã được vẽ hình chữ nhật
        rects: a list of rects.     --Danh sách hình chữ nhật
    Return:
        img_rects: an image with rects.     --Ảnh với hình chữ nhật
    """
    img_copy = img.copy()
    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(img_copy, (x,y), (x+w,y+h), (0,255,0), 2) #Vẽ hình chữ nhật với các tham số
                                                                # (Ảnh cần vẽ, góc trên cùng bên trái, góc dưới cùng bên phải, màu , độ dày)
    return img_copy



def hog_extra_and_svm_class(proposal, clf, resize = (64, 64)):
    """classify the region proposal.    --Phân loại khu vực đề xuất
    Args:
        proposal: region proposal (numpy array).    --Khu vực đề xuất
        clf: a SVM model.   --Mẫu SVM
        resize: resize the region proposal      --chỉnh kích thước khu vực đề xuất
            (default: (64, 64))
    Return:
        cls_prop: propabality of all classes.
    """
    img = cv2.cvtColor(proposal, cv2.COLOR_BGR2GRAY)    #chuyển sang ảnh xám
    img = cv2.resize(img, resize)   #chỉnh kích thước ảnh
    bins = 9
    cell_size = (8, 8)
    cpb = (2, 2)
    norm = "L2"
    features = ft.hog(img, orientations=bins, pixels_per_cell=cell_size, 
                        cells_per_block=cpb, block_norm=norm, transform_sqrt=True)
    print ("feature = ", features.shape)
    features = np.reshape(features, (1,-1))
    cls_prop = clf.predict_proba(features)
    print("type = ", cls_prop)
    print ("cls prop = ", cls_prop)
    return cls_prop

def formatConvert(file_name):
    new_file_name = file_name[:-((len(file_name))-(file_name.rindex('.')))]
    return new_file_name

if __name__ == "__main__":
    img_name = "IMG_2988.png"
    img = cv2.imread("./input/phone/"+ img_name)
    height, width, depth = img.shape
    img = cv2.copyMakeBorder(img,35,36,30,31,cv2.BORDER_CONSTANT,value=[255,255,255])
    W = 700
    imgScale = W / width
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
    img = cv2.resize(img, (int(newX), int(newY)))
    img_bin = preprocess_img(img,False)
    cv2.imshow("bin image", img_bin)
    cv2.imwrite("./output/output_images/bin_"+ img_name, img_bin)
    print(formatConvert(img_name))
    min_area = img_bin.shape[0]*img.shape[1]/(25*25)
    rects = contour_detect(img_bin, min_area=min_area)
    img_rects = draw_rects_on_img(img, rects)
    cv2.imshow("image with rects", img_rects)
    cv2.imwrite("./output/output_images/rects_"+ img_name, img_rects)

    clf = joblib.load("./svm_hog_classification/svm_model.pkl")

    img_bbx = img.copy()

    for rect in rects:
        xc = int(rect[0] + rect[2]/2)
        yc = int(rect[1] + rect[3]/2)

        size = max(rect[2], rect[3])
        x1 = max(0, int(xc-size/2))
        y1 = max(0, int(yc-size/2))
        x2 = min(width, int(xc+size/2))
        y2 = min(height, int(yc+size/2))
        proposal = img[y1:y2, x1:x2]
        cls_prop = hog_extra_and_svm_class(proposal, clf)
        cls_prop = np.round(cls_prop, 2)[0]
        cls_num = np.argmax(cls_prop)
        cls_name = cls_names[cls_num]
        prop = cls_prop[cls_num]
        if cls_name is not "background":
            cv2.rectangle(img_bbx,(rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0,0,255), 2)
            cv2.putText(img_bbx, cls_name+str(prop), (rect[0], rect[1]), 1, 1, (0,0,255),2)

    cv2.imshow("detect result", img_bbx)
    cv2.imwrite("./output/output_images/detect_"+ img_name, img_bbx)
    cv2.waitKey(0)
