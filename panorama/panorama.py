import cv2
import easygui
import numpy as np

def normalize(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts)
    scale = np.sqrt(2) / std
    T = np.array([[scale, 0, -scale * mean[0]],
                  [0, scale, -scale * mean[1]],
                  [0, 0, 1]])
    pts_normalized = np.dot(T, np.vstack((pts.T, np.ones(pts.shape[0]))))

    return pts_normalized[:2].T, T

def DLT(src_pts, dst_pts):
    src_pts_normalized, T_src = normalize(src_pts)
    dst_pts_normalized, T_dst = normalize(dst_pts)

    A = []
    for i in range(len(src_pts)):
        x, y = src_pts_normalized[i]
        x_prime, y_prime = dst_pts_normalized[i]

        row1 = [-x, -y, -1,  0,  0,  0, x * x_prime, y * x_prime, x_prime]
        row2 = [ 0,  0,  0, -x, -y, -1, x * y_prime, y * y_prime, y_prime]

        A.append(row1)
        A.append(row2)

    A = np.array(A)

    U, S, Vt = np.linalg.svd(A)
    H_normalized = Vt[-1].reshape(3, 3)
    H_normalized /= H_normalized[2, 2]
    H = np.dot(np.linalg.inv(T_dst), np.dot(H_normalized, T_src))
    H /= H[2, 2]

    return H

def panorama(img1, img2):

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT.create()
    keypoints1, descriptors1 = sift.detectAndCompute(img2_gray, mask=None)
    keypoints2, descriptors2 = sift.detectAndCompute(img1_gray, mask=None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    if len(matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches])  # Izvlačenje (x, y) koordinata
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches])  # Izvlačenje (x, y) koordinata

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4)
        if mask is not None:
            src_pts = src_pts[mask.ravel() == 1]
            dst_pts = dst_pts[mask.ravel() == 1]

        homography = DLT(src_pts, dst_pts)

    width = img2.shape[1] + img1.shape[1]
    height = img2.shape[0] + img1.shape[0]

    img_final = cv2.warpPerspective(img2, homography, (width, height))
    img_final[0:img1.shape[0], 0:img1.shape[1]] = img1

    cv2.imshow('panorama', img_final)

if __name__=="__main__":
    code1 = easygui.fileopenbox()
    img1 = cv2.imread(code1, 1)
    cv2.imshow('slika1', img1)
    cv2.moveWindow('slika1', 200, 200)

    code2 = easygui.fileopenbox()
    img2 = cv2.imread(code2, 1)
    cv2.imshow('slika2', img2)

    panorama(img1, img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()