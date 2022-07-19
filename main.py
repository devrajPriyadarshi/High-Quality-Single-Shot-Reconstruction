import logging
import numpy as np
import cv2 as cv
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
    print("")

def segmentFace(img):
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

def getRecktified(img1, img2):
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
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
        if m.distance < 0.4*n.distance:
            matchesMask[i] = [1, 0]
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    
    draw_params = dict( matchColor=(0, 255, 0),
                        singlePointColor=(255, 0, 0),
                        matchesMask=matchesMask[:],
                        flags=cv.DrawMatchesFlags_DEFAULT)

    keypoint_matches = cv.drawMatchesKnn( img1, kp1, img2, kp2, matches[:], None, **draw_params)
    showImg(scale(keypoint_matches, 0.2, 0.2))
    

if __name__ == "__main__":

    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG)

    # showImg(scale(img1, 0.1, 0.1))
    print("CAMERA PARAMETERS:\n")
    showCameraParameters()

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
    # showImg(seg1)
    # showImg(scale(seg1, 0.3, 0.3))
    logging.info("Starting Rectification!")
    getRecktified(seg1, seg2)
    logging.info("Rectification Done!\n")
