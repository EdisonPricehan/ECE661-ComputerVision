import numpy as np
import cv2
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # read books images
    books1 = cv2.imread('/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/books_1.jpeg',
                        cv2.IMREAD_GRAYSCALE)
    books2 = cv2.imread('/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/books_2.jpeg',
                        cv2.IMREAD_GRAYSCALE)

    # read fountain images
    fountain1 = cv2.imread('/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/fountain_1.jpg',
                        cv2.IMREAD_GRAYSCALE)
    fountain2 = cv2.imread('/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/fountain_2.jpg',
                        cv2.IMREAD_GRAYSCALE)

    # select input image pair
    # img1, img2 = books1, books2
    img1, img2 = fountain1, fountain2

    # Initiate SURF detector with Hessian Threshold
    surf = cv2.xfeatures2d.SURF_create(400)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(img1, None)
    print(f"Found {len(kp1)} keypoints in image1.")
    kp2, des2 = surf.detectAndCompute(img2, None)
    print(f"Found {len(kp2)} keypoints in image2.")

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    print(f"Established {len(matches)} point correspondences.")

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    matches_len = len([1 for match in matchesMask if match[0] == 1])
    print(f"Enabled {matches_len} matches.")

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    plt.imshow(img3,), plt.show()