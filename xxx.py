#coding=utf-8
#
# 图像像素访问.通道分离与合并
#

import math
import cv2
import numpy as np
#import mean()
#import r2i

def log_trans(x,k,t):#x为输入图像，k为非线性强度，t为亮度分界线
    #return ((2.0*t+2*t/50)/k)*np.log((2.0*t+t/50-x)/(x+t/50))+t
	return ((2.0*t+2*t/50)/(((2.0*t+2*t/50)/(-t))*np.log((2*k*t+t/50)/(t/50))))*np.log((2*k*t+t/50-k*x)/(k*x+t/50))+t
#def log_trans(x,k1,t):#x为输入图像，k为非线性强度，t为亮度分界线
    #return ((2.0*t+2*t/50)/k)*np.log((2.0*t+t/50-x)/(x+t/50))+t
#	return ((2.0*t+2*t/50)/(((2.0*t+2*t/50)/(-t))*np.log((2*k1*t+t/50)/(t/50))))*np.log((2*k1*t+t/50-k1*x)/(k1*x+t/50))+t
def log_3x(x,k):#x为输入图像，k为非线性强度，t为亮度分界线
    return ((255)/np.log(k*255+1))*np.log(k*x+1)

	
#k=-9.41485
k = 9.8


img_dir = "02.JPG"

    
img = cv2.imread(img_dir)

b = np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)  
g = np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)
r = np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)


#分离方法1：使用OpenCV自带的split函数
b, g, r = cv2.split(img)
img_gary = cv2.imread(img_dir,0)

hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
h = np.zeros((hsv_img.shape[0],hsv_img.shape[1]),dtype=hsv_img.dtype)  
s = np.zeros((hsv_img.shape[0],hsv_img.shape[1]),dtype=hsv_img.dtype)
v = np.zeros((hsv_img.shape[0],hsv_img.shape[1]),dtype=hsv_img.dtype)
h, s, v = cv2.split(hsv_img)
print("sbhhshhdhs=",np.mean(v))
if np.mean(v)<45:
    v=v*1.2
cv2.imwrite("v18.jpg",v)
v = cv2.imread("v18.jpg",0) 

tv, img_otsu_v = cv2.threshold(v, 0, 255, cv2.THRESH_OTSU)
#img_otsu_ad = cv2.adaptiveThreshold(v,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,1111,0)

t, img_otsu_ga = cv2.threshold(img_gary, 0, 255, cv2.THRESH_OTSU)
tb, img_otsu_b = cv2.threshold(b, 0, 255, cv2.THRESH_OTSU)
tg, img_otsu_g = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)
tr, img_otsu_r = cv2.threshold(r, 0, 255, cv2.THRESH_OTSU)


##取最大证明非逆光
img_otsu_yz1 = np.dstack([img_otsu_b,img_otsu_g,img_otsu_r, img_otsu_v,img_otsu_ga])
print(img_otsu_yz1.shape)
img_otsu_yz = img_otsu_yz1.max(axis=-1)

img_otsu_b = img_otsu_b
img_otsu_g = img_otsu_g
img_otsu_r = img_otsu_r
cv2.imwrite("yz.jpg",img_otsu_yz)

b1=np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)
b2=np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)
b3=np.zeros((img.shape[0],img.shape[1]))
b_temp=np.zeros((img.shape[0],img.shape[1]))
b_otsu_255=np.ones((img.shape[0],img.shape[1]))
b_otsu_255=b_otsu_255*255
b_otsu_255=b_otsu_255-img_otsu_b

b1=(img_otsu_b/255)*b
b2=(b_otsu_255/255)*b


print(tb, img_otsu_b)
cv2.imwrite("r.jpg",img_otsu_r)
cv2.imwrite("g.jpg",img_otsu_g)
cv2.imwrite("ga.jpg",img_otsu_ga)
cv2.imwrite("b.jpg",img_otsu_b)
cv2.imwrite("v.jpg",img_otsu_v)
#cv2.imwrite("ad.jpg",img_otsu_ad)
cv2.imwrite("v-r.jpg",img_otsu_v-img_otsu_r)
cv2.imwrite("v-g.jpg",img_otsu_v-img_otsu_g)
cv2.imwrite("v-b.jpg",img_otsu_v-img_otsu_b)

g1=np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)
g2=np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)
g3=np.zeros((img.shape[0],img.shape[1]))
g_temp=np.zeros((img.shape[0],img.shape[1]))
g_otsu_255=np.ones((img.shape[0],img.shape[1]))
g_otsu_255=g_otsu_255*255
g_otsu_255=g_otsu_255-img_otsu_g

g1=(img_otsu_g/255)*g
g2=(g_otsu_255/255)*g



r1=np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)
r2=np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)
r3=np.zeros((img.shape[0],img.shape[1]))
r_temp=np.zeros((img.shape[0],img.shape[1]))
r_otsu_255=np.ones((img.shape[0],img.shape[1]))
r_otsu_255=r_otsu_255*255
r_otsu_255=r_otsu_255-img_otsu_r

r1=(img_otsu_r/255)*r
r2=(r_otsu_255/255)*r

b2=log_trans(b2,k,tb)
g2=log_trans(g2,k,tg)
r2=log_trans(r2,k,tr)

b3=b1+b2
g3=g1+g2
r3=r1+r2
img_updata_bgr = np.dstack([b3,g3,r3])
cv2.imwrite("img_updata_bgr.jpg",img_updata_bgr)



##v通道
#v=log_trans(v,k,tv)


v1=np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)
v2=np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)
v3=np.zeros((img.shape[0],img.shape[1]))
v_temp=np.zeros((img.shape[0],img.shape[1]))
v_otsu_255=np.ones((img.shape[0],img.shape[1]))
v_otsu_255=v_otsu_255*255
v_otsu_255=v_otsu_255-img_otsu_v

v1=(img_otsu_v/255)*v
v2=(v_otsu_255/255)*v
v2=log_trans(v2,1,tv)
cv2.imwrite("v1.jpg",v1)
v3=v1+v2
cv2.imwrite("v3.jpg",v3)
v3=v3.astype(hsv_img.dtype)
hsv_img_updata = np.dstack([h,s,v3])

rgb_img2 = cv2.cvtColor(hsv_img_updata,cv2.COLOR_HSV2BGR)

imga = cv2.imwrite("rgb_updata_v.jpg",rgb_img2)

##l通道
l1=np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)
l2=np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)
l3=np.zeros((img.shape[0],img.shape[1]))
l_temp=np.zeros((img.shape[0],img.shape[1]))
l_otsu_255=np.ones((img.shape[0],img.shape[1]))
l_otsu_255=l_otsu_255*255
l_otsu_255=l_otsu_255-img_otsu_yz

l1=(img_otsu_yz/255)*v
l2=(l_otsu_255/255)*v 
print("LLLLLLLLLL2=",np.mean(l2))
if np.mean(l2)<46 and np.mean(v)/np.mean(l2)<=3.8:


	if np.mean(l2)<20:
		k1 = 0.0001 + 255/np.mean(l2)
		print("ki,jpg")
	elif np.mean(l2)>20 and np.mean(l2)<30:
		k1 = 0.0001 + 165/np.mean(l2) 
	elif np.mean(l2)>30 and np.mean(l2)<40:
		k1 = 0.0001 +80/np.mean(l2) 	
	else:
		k1 = 0.0001 +20/np.mean(l2) 
	l2=log_trans(l2,k1,tv)
	l2=log_trans(l2,0.01,255)

if np.mean(l2)<46 and (np.mean(v)/np.mean(l2))>3.8 and (np.mean(v)-np.mean(l2))>100:
	print("VHJKKJ=",np.mean(v))
    #l_temp=np.zeros((img.shape[0],img.shape[1]))
	l_otsu_255=np.ones((img.shape[0],img.shape[1]))
	l_otsu_255=l_otsu_255*255
	l_otsu_255=l_otsu_255-img_otsu_v
	l1=np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)
	l2=np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)
	print(v.shape)
	l1=(img_otsu_v/255)*v
	l2=(l_otsu_255/255)*v 
	print("VVVVVVVVVVL2=",np.mean(l2))
	
	if np.mean(l2)<20 and np.mean(l2)>6:
		k1 = 0.0001 + 155/np.mean(l2)  
	elif np.mean(l2)>20 and np.mean(l2)<30:
		k1 = 0.0001 + 25/np.mean(l2) 
	elif np.mean(l2)>30 and np.mean(l2)<40:
		k1 = 0.0001 +10/np.mean(l2)
	else:
		k1 = 0.0001 + 1/np.mean(l2) 
	l2=log_trans(l2,k1,tv)
	l2=log_trans(l2,0.01,255)






cv2.imwrite("l2.jpg",l2)
l2 = cv2.imread("l2.jpg",0)  

cv2.imwrite("l1.jpg",l1)
cv2.imwrite("l2.jpg",l2)
cv2.imwrite("v.jpg",v)
l3=l1+l2
cv2.imwrite("l3.jpg",l3)
l3=l3.astype(hsv_img.dtype)
hsvl_img_updata = np.dstack([h,s,l3])

rgb_img2l = cv2.cvtColor(hsvl_img_updata,cv2.COLOR_HSV2BGR)

imga = cv2.imwrite("rgb_updata_l.jpg",rgb_img2l)


v_eq = cv2.equalizeHist(v)
eq_img_updata = np.dstack([h,s,v_eq])
eq_img2l = cv2.cvtColor(eq_img_updata,cv2.COLOR_HSV2BGR)
imga = cv2.imwrite("rgb_updata_eq.jpg",eq_img2l)


v_eq = v*3.5
cv2.imwrite("v_eq15.jpg",v_eq)
v_eq = cv2.imread("v_eq15.jpg",0)
eq_img_updata = np.dstack([h,s,v_eq])
eq_img2l = cv2.cvtColor(eq_img_updata,cv2.COLOR_HSV2BGR)
imga = cv2.imwrite("rgb_updata_beishu.jpg",eq_img2l)

v_eq = log_3x(v,1.2)
cv2.imwrite("v_feq15.jpg",v_eq)
v_eq = cv2.imread("v_feq15.jpg",0)
eq_img_updata = np.dstack([h,s,v_eq])
eq_img2l = cv2.cvtColor(eq_img_updata,cv2.COLOR_HSV2BGR)
imga = cv2.imwrite("rgb_updata_fxx.jpg",eq_img2l)









