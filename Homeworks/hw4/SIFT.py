
import cv2
import matplotlib.pyplot as plt


def sift_corr(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    print(f"{img1.shape=}")
    print(f"{img2.shape=}")

    # initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    print(f"Found {len(kp1)} keypoints in image1, {kp1[0]}.")
    kp2, des2 = sift.detectAndCompute(img2, None)
    print(f"Found {len(kp2)} keypoints in image2, {kp2[0]}.")

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    print(f"Established {len(matches)} point correspondences.")

    # need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    matches_len = len([1 for match in matchesMask if match[0] == 1])
    print(f"Enabled {matches_len} matches.")

    draw_params = dict(
                       # matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()


if __name__ == "__main__":
    # read books images
    # books1 = '/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/books_1.jpeg'
    # books2 = '/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/books_2.jpeg'
    # sift_corr(books1, books2)

    # read fountain images
    # fountain1 = '/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/fountain_1.jpg'
    # fountain2 = '/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/fountain_2.jpg'
    # sift_corr(fountain1, fountain2)

    # read building images
    # building1 = '/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/building_1.jpg'
    # building2 = '/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/building_2.jpg'
    # sift_corr(building1, building2)

    # read garden images
    garden1 = '/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/garden_1.jpg'
    garden2 = '/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/garden_2.jpg'
    sift_corr(garden1, garden2)



