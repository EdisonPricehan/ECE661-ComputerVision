
import numpy as np
import cv2
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # read books images
    books1 = cv2.imread('/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/books_1.jpeg')
    books2 = cv2.imread('/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/books_2.jpeg')

    # read fountain images
    fountain1 = cv2.imread('/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/fountain_1.jpg')
    fountain2 = cv2.imread('/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/fountain_2.jpg')

    # select input image pair, rescale and convert to gray images
    # img1, img2 = books1, books2
    img1, img2 = fountain1, fountain2
    print(f"{img1.shape=}")
    scale = 0.5
    dim1 = (int(img1.shape[1] * scale), int(img1.shape[0] * scale))
    dim2 = (int(img2.shape[1] * scale), int(img2.shape[0] * scale))
    img1 = cv2.resize(img1, dim1)
    img2 = cv2.resize(img2, dim2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray1 = np.float32(gray1)
    gray2 = np.float32(gray2)

    corners1 = cv2.cornerHarris(gray1, 2, 3, 0.04)
    corners2 = cv2.cornerHarris(gray2, 2, 3, 0.04)
    print(f"{corners1=}")
    print(f"{corners1.shape=}")
    # corners1 = cv2.dilate(corners1, None)
    # corners2 = cv2.dilate(corners2, None)

    img1[corners1 > 0.01 * corners1.max()] = [0, 0, 255]
    img2[corners2 > 0.01 * corners2.max()] = [0, 0, 255]

    img3 = np.concatenate((img1, img2), axis=1)

    plt.imshow(img3), plt.show()

    # kp1 =

    # draw_params = dict(matchColor=(0, 255, 0),
    #                    singlePointColor=(255, 0, 0),
    #                    matchesMask=None,
    #                    flags=cv2.DrawMatchesFlags_DEFAULT)
    #
    # img3 = cv2.drawMatchesKnn(img1, corners1, img2, corners2, [], None)
    # plt.imshow(img3, ), plt.show()



