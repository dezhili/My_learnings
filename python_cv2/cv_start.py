import numpy as np 
import cv2

'''
read an image   display an image   write an image
'''

# img = cv2.imread('tensorflow.jpg', 0)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()  # 自适应窗口大小


# Write an image  use cv2.imwrite() to save an image
# cv2.imwrite('tensorflow_gray.png', img)



# sum it up
# img = cv2.imread('pytorch.jpg', 0)
# cv2.imshow('image', img)
# k = cv2.waitKey(0)
# if k == 27:                        # wait for ESC key to exit
#     cv2.destroyAllWindows()
# elif k == ord('s'):                # wait for 's' key to save and exit
#     cv2.imwrite('pytorch_gray.png', img)
#     cv2.destroyAllWindows()


from matplotlib import pyplot as plt

img = cv2.imread('pytorch.jpg',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()