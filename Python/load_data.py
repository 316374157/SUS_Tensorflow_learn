import os
import sys
import numpy as np
import cv2

IMAGE_SIZE=64

#按照指定图像大小调整尺寸
#esize_image()函数。这个函数的功能是判断图片是不是正方形，
#如果不是则增加短边的长度使之变成正方形。这样再调用cv2.resize()函数就可以实现等比例缩放了。
#因为我们指定缩放的比例就是64 x 64，只有缩放之前图像为正方形才能确保图像不失真。
def resize_image(image,height=IMAGE_SIZE,width=IMAGE_SIZE):
	top,bottom,left,right=(0,0,0,0)

	#获取图像的尺寸
	h,w,_=image.shape

	#对于长宽不相等的图片，找到最长的一边
	longest_edge=max(h,w)

	#计算短边需要增加多上像素宽度使其与长边相等
	if h<longest_edge:
		dh=longest_edge - h
		top=dh//2
		bottom=dh - top
	elif w<longest_edge:
		dw=longest_edge - w
		left=dw//2
		right=dw - left
	else:
		pass

	#RGB颜色
	BLACK=[0,0,0]

	#给图像增加边界，是图片长，宽等长，BORDER_CONSTANT指定边界颜色由value指定
	constant=cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=BLACK)

	#调整图像大小并返回
	return cv2.resize(constant,(height,width))

#读取训练数据
images=[]
labels=[]
def read_path(path_name):
	for dir_item in os.listdir(path_name):
		#从初始路径开始叠加，合并成可识别的操作路径
		full_path=os.path.abspath(os.path.join(path_name,dir_item))

		if os.path.isdir(full_path):        #如果是文件夹，继续递归调用
			read_path(full_path)
		else:    #文件
			if dir_item.endswith('.jpg'):
				image=cv2.imread(full_path)
				image=resize_image(image,IMAGE_SIZE,IMAGE_SIZE)

				#放开这个代码，可以看到resize_image()函数的实际调用效果
				#cv2.imwrite('1.jpg',image)

				images.append(image)
				labels.append(path_name)
	return images,labels

#从指定路径读取训练数据集
def load_datatest(path_name):
	images,labels=read_path(path_name)

	#将输入的所有图片转成四维数组，尺寸为（图片数量*IMAGE_SIZE*IMAGE_SIZE*3)
	#例我和你两个人共1200张图片，IMAGE_SIZE为64，故对我来说尺寸为1200*64*64*3
	#图片为64*64像素，一个像素3个颜色值（RGB）
	images=np.array(images)
	print(images.shape)

	#标注数据,''是文件夹名字放置同一个人的脸部图像，全部指定为0，令一个文件夹下是其他人
	labels = np.array([0 if label.endswith('cv_image') else 1 for label in labels])
	return images,labels

if __name__=='__main__':
	if len(sys.argv)!=1:
		print('Uage:%s path_name\r\n' % (sys.argv[0]))
	else:
		images,labels=load_datatest('F:/Tensorflow-DeepLearning/Image_data')
