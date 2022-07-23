import logging
from math import log
import numpy as np
from numpy.linalg import inv
import cv2 as cv
from matplotlib import pyplot as plt
# import open3d 

img1 = cv.imread('Files/View1.png', 1)
img2 = cv.imread('Files/View2.png', 1)

K = [   [4876.8,    0,      2032],
        [0,         4876.8, 1520],
        [0,         0,      1   ]]

R1 = [  [0.990721,      0.00393774,     -0.135854   ],
		[-0.00406891,   0.999991,       0.000687881 ],
		[0.13585,       -0.00123427,    0.990729    ]]

t1 = [  [-0.639768  ],
		[-0.00425937],
		[-0.0355112 ]]

R2 = [  [0.998179,      0.00217893,         -0.0602765  ],
		[-0.00220042,   0.9999980000000001, 0.000290223 ],
		[0.0602758,     -0.000422328,       0.998182    ]]

t2 = [  [-0.267663  ],
		[-0.00128758],
		[-0.0993487 ]]

K = np.array(K)
R1 = np.array(R1)
t1 = np.array(t1)
R2 = np.array(R2)
t2 = np.array(t2)
P1 = K @ np.array(np.bmat([R1,t1]))
P2 = K @ np.array(np.bmat([R2,t2]))

p = np.array(np.bmat([[inv(K)],[np.zeros((1,3))]]))
c = np.array([[0],[0],[0],[1]])
# F = np.array(np.cross(P2, c)@P2@p)

P2_ = P2.T @ inv(P2@P2.T)
# F2 = np.array(np.cross(P1, _) @ P1 @ P2_)

def scale(img, sx, sy):
    # print("Image is of ", img.shape)
    sclImg = cv.resize(img, None, fx = sx, fy = sy)
    # print("Scaling to", sclImg.shape, "!\n")
    return sclImg

def showImg(img):
    # print("Size of Img = ", img.shape)
    cv.imshow("IMG | Press any key to exit window", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def showCameraParameters():
    print("Intrinsic Matrix (K) =\n", K,"\n")
    print("R1 =\n", R1,"\n")
    print("t1 =\n", t1,"\n")
    print("R2 =\n", R2,"\n")
    print("t2 =\n", t2,"\n")
    print("P1 =\n", P1,"\n")
    print("P2 =\n", P2,"\n")
    print("p =\n", p,"\n")
    print("c =\n", c,"\n")
    # print("F =\n", F,"\n")

    print("")

def segmentFace(img):

    """Todo:REPLEACE THIS WITH COLOR-BASED SEGMENTATION"""

    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    # rect = (640,0,2760,img.shape[1]) #when s = 1
    rect = (64,0,276,img.shape[1]) # when s = 0.1
    cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask==2)|(mask==0),0,1).astype(np.uint8)
    img = img*mask2[:,:,np.newaxis]
    # showImg(segmentFace(scale(img1, 0.1, 0.1))[0])
    # showImg(segmentFace(scale(img2, 0.1, 0.1))[0])
    return img, mask2

def getSegmentedImgs():
    maskImg1 = scale(segmentFace(scale(img1, 0.1, 0.1))[1], 10.01, 10)
    img1Seg = img1*maskImg1[:,:,np.newaxis]
    maskImg2 = scale(segmentFace(scale(img2, 0.1, 0.1))[1], 10.01, 10)
    img2Seg = img2*maskImg2[:,:,np.newaxis]
    # showImg(scale(img1Seg, 0.1, 0.1))
    # showImg(scale(img2Seg, 0.1, 0.1))
    return img1Seg, img2Seg

def getSegmentedImgs2(img1=img1, img2=img2):
    img1Seg = segmentFace(img1)[0]
    img2Seg = segmentFace(img2)[0]
    
    return img1Seg, img2Seg

def segmentSkin(img1, img2):
    low_skin = np.array([69, 60, 90])
    high_skin = np.array([195, 170, 255])
    mask1 = cv.inRange(img1, low_skin, high_skin)
    mask2 = cv.inRange(img2, low_skin, high_skin)
    seg1 = cv.bitwise_and(img1, img1, mask = mask1)
    seg2 = cv.bitwise_and(img2, img2, mask = mask2)
    showImg(scale(seg1, 0.3, 0.3))
    showImg(scale(seg2, 0.3, 0.3))

def drawlines(img1src, img2src, lines, pts1src, pts2src):
    r, c, h = img1src.shape
    img1color = img1src
    img2color = img2src
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv.circle(img2color, tuple(pt2), 5, color, -1)
    return img1color, img2color

def getRecktified(img1, img2):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # imgSift = cv.drawKeypoints(img1, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # showImg(scale(imgSift,0.3,0.3))
    # imgSift = cv.drawKeypoints(img2, kp2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # showImg(scale(imgSift,0.3,0.3))
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    matchesMask = [[0, 0] for i in range(len(matches))]
    good = []
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75*n.distance:
            matchesMask[i] = [1, 0]
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    
    # draw_params = dict( matchColor=(0, 255, 0),
                        # singlePointColor=(255, 0, 0),
                        # matchesMask=matchesMask[:],
                        # flags=cv.DrawMatchesFlags_DEFAULT)

    # keypoint_matches = cv.drawMatchesKnn( img1, kp1, img2, kp2, matches[:], None, **draw_params)
    # showImg(scale(keypoint_matches, 0.2, 0.2))

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    fundamental_matrix, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

    print("funda =\n",fundamental_matrix,"\n")
    pts1 = pts1[inliers.ravel() == 1]
    pts2 = pts2[inliers.ravel() == 1]

    # lines1 = cv.computeCorrespondEpilines(
    # pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
    # lines1 = lines1.reshape(-1, 3)
    # img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    # # Find epilines corresponding to points in left image (first image) and
    # # drawing its lines on right image
    # lines2 = cv.computeCorrespondEpilines(
    #     pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
    # lines2 = lines2.reshape(-1, 3)
    # img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    # plt.subplot(211), plt.imshow(img5)
    # plt.subplot(212), plt.imshow(img3)
    # plt.suptitle("Epilines in both images")
    # plt.show()

    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape
    _, H1, H2 = cv.stereoRectifyUncalibrated (np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1))
    img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))
    cv.imwrite("Files/rectified_1.png", img1_rectified)
    cv.imwrite("Files/rectified_2.png", img2_rectified)
    # showImg(scale(img1_rectified, 0.3, 0.3))
    # showImg(scale(img2_rectified, 0.3, 0.3))
    return img1_rectified, img2_rectified

def subsample(img1, img2):
    rows, cols, _channels = img1.shape

    subImg1 = cv.pyrDown(img1,dstsize=(cols//2, rows//2))
    subImg2 = cv.pyrDown(img2,dstsize=(cols//2, rows//2))
    
    return subImg1, subImg2
    # showImg(img1)


if __name__ == "__main__":

    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG)

    # showImg(scale(img1, 0.3, 0.3))
    print("CAMERA PARAMETERS:\n")
    showCameraParameters()

    """START RECTIFICATION"""
    logging.info("Starting Face Segementation from Background:")
    seg1 = cv.imread("Files/Segmented1.png", 1)
    seg2 = cv.imread("Files/Segmented2.png", 1)
    try:
        if seg1 == None or seg2 == None:
            logging.warning("No Saved Images Found. Will continue segmentation process.")
            logging.warning("This will take a long time.")
            seg1, seg2 = getSegmentedImgs()
    except ValueError:
        logging.info("FOUND SAVED SEGEMENTED IMAGES! Skipping Segementation")
    logging.info("Segmentation Done!\n")

    """START RECTIFICATION"""
    logging.info("Starting Rectification!")
    rect1 = cv.imread("Files/rectified_1.png", 1)
    rect2 = cv.imread("Files/rectified_2.png", 1)
    try:
        if rect1 == None or rect2 == None:
            logging.warning("No Saved Images Found. Will continue Rectification process.")
            logging.warning("This will take not much time.")
            rect1, rect2 = getRecktified(seg1, seg2)
    except ValueError:
        logging.info("FOUND SAVED RECTIFIED IMAGES! Skipping Rectification")
    logging.info("Rectification Done!\n")

    """START BUILDING A IMAGE PYRAMID"""
    logging.info("Process Image Pyramid!")
    pyramidImg1 = [rect1]
    pyramidImg2 = [rect2]
    for i in range( int(log(rect1.shape[0]/150, 2))):
        sub1, sub2 = subsample(pyramidImg1[0], pyramidImg2[0])
        pyramidImg1 = [sub1] + pyramidImg1
        pyramidImg2 = [sub2] + pyramidImg2

    # for x in pyramidImg2:
    #     showImg(x)
    logging.info("Image Pyramid Made!")

    logging.info("Calculate Disparity Map at each level!")

    