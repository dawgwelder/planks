import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

pic_pth = 'test4.png'

img = cv2.imread(pic_pth)
#cv2.rectangle(img, (210, 170), (300, 400), (0, 255, 0), 3)
img2 = img[170:400, 210:300]
gray = cv2.imread(pic_pth, 0)
sobely = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=-1)
th2 = cv2.inRange(sobely, 100, 255)
sobely = cv2.Canny(gray,150,255, 10)
kernel = np.ones((3,3),np.uint8)
sobely = cv2.dilate(sobely,kernel,iterations = 2)

        # cv2.circle(img,(x1,y1),5,(255,0, 0),5)
        #print((x1,y1))
# ret2,th2 = cv2.threshold(sobely,240,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#min_pix = np.asarray ([img2[..., 0].min(), img2[..., 1].min(), img2[..., 2].min()]) - 15
#max_pix = np.asarray([ img2[..., 0].max(), img2[..., 1].max(), img2[..., 2].max()]) + 15
# min_pix = np.array ((79, 93, 103))
# max_pix = np.array((159, 161, 165))
# print (min_pix, max_pix)
# mask = cv2.inRange(img, min_pix, max_pix)
kernel = np.ones((3,3),np.uint8)

lines = cv2.HoughLines(sobely,1,np.pi/2,200)
print (len(lines))
y_1 = 0
y_2 = 10000
for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        y_1 = max(y_1, y1)
        y_2 = min(y_2, y2)
        # cv2.line(mask,(x1,y1),(x2,y2),(255,255,255),3)
        #cv2.line(img,(x1,y1),(x2,y2),(255,255,255),1)
#rect = cv2.boundingRect(y1,y2)
img3 =img[y_2:y_1, :]
av_pix = np.asarray ([img3[..., 0].mean(), img3[..., 1].mean(), img3[..., 2].mean()])
min_pix = av_pix - 20
max_pix = av_pix + 50
mask = cv2.inRange(img3, min_pix, max_pix)
print (av_pix)
height, width, channels = img.shape
height1, width1, channels1 = img3.shape
cv2.line(mask,(0,0),(width1, 0),(255,255,255),3)
cv2.line(mask,(0,height1),(width1,height1),(255,255,255),3)
cv2.line(mask,(0,0),(0,height1),(255,255,255),3)
cv2.line(mask,(width1, 0),(width1, height1),(255,255,255),3)
erosion = cv2.dilate(mask, kernel, iterations=4)
er = erosion.copy()
contours = cv2.findContours(er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


cn = contours[1]
cn1 = []

arp = width1*height1
print (arp)
# print('imgsize', width,height)
for cnt in cn:
    ar = cv2.contourArea(cnt)
    per = cv2.arcLength(cnt,True)
    # print (per)
    # M = cv2.moments(cnt)
    # cx = int(M['m10'] / M['m00'])
    # cy = int(M['m01'] / M['m00'])
    # print(abs(width-cx) )
    #if  per < width*0.5:
    if ar + per*3 < arp and ar > 60:
        cn1.append(cnt)
        x = ar + per*3
        print(x)
cv2.drawContours(img3, cn1, -1, (0, 255, 0), 2)


#img1 = cv2.blur(img,(13, 13))
# ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,11,3)
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,3)
titles = ['img', 'mask', 'er', 'sobel']
images = [img3, mask, erosion] #sobely]#, th1, th2, th3]
for i in range(3):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()