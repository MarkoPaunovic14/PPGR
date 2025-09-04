import cv2
import easygui
import numpy as np
import matplotlib.pyplot as plt


# Added normalization to improve homography calculation beacuse I got streched out result
def normalize_points(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts)
    scale = np.sqrt(2) / std
    T = np.array([[scale, 0, -scale * mean[0]],
                  [0, scale, -scale * mean[1]],
                  [0, 0, 1]])
    pts_normalized = np.dot(T, np.vstack((pts.T, np.ones(pts.shape[0]))))
    return pts_normalized[:2].T, T

def compute_homography(src_pts, dst_pts):
    # Fix for streched out image...
    src_pts_normalized, T_src = normalize_points(src_pts)
    dst_pts_normalized, T_dst = normalize_points(dst_pts)

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
    
    # testing for showing matches beacuse of the error I got before normalization
    # debug_img = cv2.drawMatches(img2, keypoints1, img1, keypoints2, matches[:10], None, flags=2)
    # plt.imshow(debug_img)
    # plt.title("Top Matches")
    # plt.show()

    keypoints1 = np.float32([kp.pt for kp in keypoints1])
    keypoints2 = np.float32([kp.pt for kp in keypoints2])

    if len(matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx] for m in matches])
        dst_pts = np.float32([keypoints2[m.trainIdx] for m in matches])

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4)
        if mask is not None:
            src_pts = src_pts[mask.ravel() == 1]
            dst_pts = dst_pts[mask.ravel() == 1]

        homography = compute_homography(src_pts, dst_pts)

    width = img2.shape[1] + img1.shape[1]
    height = img2.shape[0] + img1.shape[0]

    img_final = cv2.warpPerspective(img2, homography, (width, height))
    img_final[0:img1.shape[0], 0:img1.shape[1]] = img1

    img_final_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)

    non_black_mask = np.any(img_final != 0, axis=-1)

    non_black_coords = np.column_stack(np.where(non_black_mask))

    top_left = non_black_coords.min(axis=0)
    bottom_right = non_black_coords.max(axis=0)

    img_cropped = img_final[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]

    plt.imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
    plt.title("Panorama Result")
    plt.show()

if __name__ == "__main__":
    code1 = easygui.fileopenbox(title="Select the first image")
    img1 = cv2.imread(code1, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title("Image 1")
    plt.show()

    code2 = easygui.fileopenbox(title="Select the second image")
    img2 = cv2.imread(code2, 1)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title("Image 2")
    plt.show()

    panorama(img1, img2)
