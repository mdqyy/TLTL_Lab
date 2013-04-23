#!/usr/bin/python
# This is a standalone program. Pass an image name as a first parameter of the program.

import sys
from math import sin, cos, sqrt, pi
import cv
import PyQt4
from PyQt4.QtTest import *
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from time import sleep
import math

# toggle between CV_HOUGH_STANDARD and CV_HOUGH_PROBILISTIC


class line(QWidget):
    def __init__(self, point1, point2):
        self.p1 = point1
        self.p2 = point2

    def paintEvent(self,event):
        painter=QPainter()
        painter.begin(self)
        painter.setPen(QPen(Qt.darkGray,3))
        painter.drawLine(self.p1,self.p2)
        painter.end()

class applicationWindow(QMainWindow):

	def __init__(self, frame_num = 0, parent = None, params = None):
		QMainWindow.__init__(self, parent)
		self.image_changed = False
		self.do_hough = True
		self.capture = cv.CaptureFromFile("933_5214.MOV")
		self.num_frames = int(cv.GetCaptureProperty(self.capture,7))
		use_image = False
		self.USE_STANDARD = False
		self.corner_count = 1000
		self.setAttribute(Qt.WA_PaintOutsidePaintEvent)
		frame_rate= cv.GetCaptureProperty(self.capture, 5)
		desired_frame = int(frame_num * frame_rate)
		cv.SetCaptureProperty(self.capture, 1, desired_frame)
		'''
		while curr_frame < desired_frame:
			src = cv.QueryFrame(self.capture)
			curr_frame+=1
		'''
#		self.num_frames-=desired_frame
		while not(use_image):
			get_response, ok = QInputDialog.getItem(None, "Use This Frame?", "Use This Frame?", ["No", "Yes"], 0, False)
			if ok and get_response == "Yes":
				use_image = True
			self.get_image()
		cv.SaveImage('src_image.png',self.img)
		self.image = QImage('src_image.png')
		self.resize(self.image.size())
		self.rectangles = []
		self.capturing = False
		self.good_rect_features = []
		self.rect_features_all = {} #put the circle, line and good_feature items in this structure
		self.rect_feature_assignment = {}
		self.tracked_feature_displacement = {}

	def get_image(self):
		self.image_changed = True
		height_scale = QDesktopWidget().screenGeometry().height()/cv.GetCaptureProperty(self.capture,4)
		width_scale = QDesktopWidget().screenGeometry().width()/cv.GetCaptureProperty(self.capture,3)
		scaling_factor = min(height_scale, width_scale)
		#optimize scaling factor for screen resolt
		src = cv.QueryFrame(self.capture)
		self.img_height = int(src.height * scaling_factor)
		self.img_width = int(src.width *scaling_factor)
		self.img = cv.CreateMat(self.img_height, self.img_width,cv.CV_8UC3 )
		cv.Resize(src, self.img)
		#cv.CvtScale(src,self.img,0.1)

#		self.num_frames-=1
		cv.ShowImage("Frame",self.img)

	def paintEvent(self, event):
		p = QPainter(self)
		if self.image_changed:
			p.drawImage(event.rect(), self.image)
  
	def mouseReleaseEvent(self, ev):
		if not(self.capturing):
			self.p = QPainter()
			self.p.begin(self)
			c_rect = QRect(self.currentPos, ev.pos())
			self.p.drawRect(QRect(self.currentPos, ev.pos()))
			self.rect_features_all[c_rect] = {"circles":[], "features":[], "lines":[], "displacement":[]}

	def mousePressEvent(self, ev):
		self.currentPos=QPoint(ev.pos())
	
	def mouseDoubleClickEvent(self, ev):
		if not(self.capturing):
			self.get_good_features()

	def resolve_features_to_rectangles(self, rectangle, features, circles, lines):
		for feature in features:
			if rectangle.contains(QPoint(feature[0], feature[1])) and feature not in self.good_rect_features:
				self.rect_features_all[rectangle]["features"].append(len(self.good_rect_features))
				self.good_rect_features.append(feature)
		for line in lines:
			if rectangle.contains(QPoint(line[0][0], line[0][1])) and rectangle.contains(QPoint(line[1][0], line[1][1])) and (line not in self.rect_features_all[rectangle]["lines"]):
				self.rect_features_all[rectangle]["lines"].append(line)
		for a in range(circles.rows):
			c_circle = circles[a, 0]
			c_radius = int(c_circle[2])
			c_x = int(c_circle[0])
			c_y = int(c_circle[1])
			if rectangle.contains(QPoint(c_x, c_y)) and (c_circle not in self.rect_features_all[rectangle]["circles"]):
				self.rect_features_all[rectangle]["circles"].append(c_circle)

	def get_good_features(self):
		if not self.good_rect_features:
			gray = cv.CreateImage(cv.GetSize(self.img), 8, 1)
			cv.CvtColor(self.img, gray, cv.CV_BGR2GRAY)
			self.goodFeatures = self.get_good_features_during_capture(self.img, gray)
			#restrict good features to those that are in self.rectangles
			lines, circles = self.get_lines_and_circles(self.img, gray)
			for rectangle in self.rect_features_all.keys():
				self.resolve_features_to_rectangles(rectangle, self.goodFeatures, circles, lines)
			'''
			for rectangle in self.rect_features_all.keys():
				for feature in self.goodFeatures:
					if rectangle.contains(QPoint(feature[0], feature[1])) and feature not in self.good_rect_features:
						self.rect_features_all[rectangle]["features"].append(len(self.good_rect_features))
						self.good_rect_features.append(feature)
				for line in lines:
					if rectangle.contains(QPoint(line[0][0], line[0][1])) and rectangle.contains(QPoint(line[1][0], line[1][1])):
						self.rect_features_all[rectangle]["lines"].append(line)
				for a in range(circles.rows):
					c_circle = circles[a, 0]
					c_radius = int(c_circle[2])
					c_x = int(c_circle[0])
					c_y = int(c_circle[1])
					if rectangle.contains(QPoint(c_x, c_y)):
						self.rect_features_all[rectangle]["circles"].append(circles)
			'''
			self.capture_and_process()

	def get_good_features_during_capture(self, img, gray):
		o_image = cv.CreateImage(cv.GetSize(img), 8,1)
		eig_image = cv.CreateImage(cv.GetSize(gray), cv.IPL_DEPTH_32F, 1)
		temp_image = cv.CreateImage(cv.GetSize(gray), cv.IPL_DEPTH_32F, 1)
		n_goodFeatures = cv.GoodFeaturesToTrack(gray, eig_image, temp_image, self.corner_count, 0.004, 0.01, None, 3 , False, 0.04)
		n_goodFeatures = cv.FindCornerSubPix(o_image,n_goodFeatures,(5,5),(-1,-1),(cv.CV_TERMCRIT_EPS, 0, 0.01))
		return n_goodFeatures
	
	def get_lines_and_circles(self, src, gray):
		smooth  = cv.CreateImage(cv.GetSize(src), 8, 1)
		cv.CvtColor(src, gray, cv.CV_BGR2GRAY)
		dst = cv.CreateImage(cv.GetSize(src), 8, 1)
		#dst2 = cv.CreateImage(cv.GetSize(src), cv.IPL_DEPTH_16S, 3)
		cv.Smooth(gray, smooth, cv.CV_GAUSSIAN)
		#cv.Laplace(src, dst2)		
		color_dst = cv.CreateImage(cv.GetSize(src), 8, 3)
		storage = cv.CreateMemStorage(0)
		circles = cv.CreateMat(int(src.height * src.width), 1, cv.CV_32FC3)
		lines = 0
		cv.Canny(gray, dst, 50, 200, 3)
		cv.CvtColor(dst, color_dst, cv.CV_GRAY2BGR)
		
		if self.do_hough:
			#associate lines with specific rectangles in the first frame
			if self.USE_STANDARD:
				lines = cv.HoughLines2(dst, storage, cv.CV_HOUGH_STANDARD, 1, pi / 180, 10, 0, 0)
				for (rho, theta) in lines[:100]:
					a = cos(theta)
					b = sin(theta)
					x0 = a * rho 
					y0 = b * rho
					pt1 = (cv.Round(x0 + 1000*(-b)), cv.Round(y0 + 1000*(a)))
					pt2 = (cv.Round(x0 - 1000*(-b)), cv.Round(y0 - 1000*(a)))
					cv.Line(src, pt1, pt2, cv.RGB(255, 0, 0), 3, 8)
			else:
				lines = cv.HoughLines2(dst, storage, cv.CV_HOUGH_PROBABILISTIC, 1, pi / 180, 10, 100, 10)
				#lines = cv.HoughLines2(dst, storage, cv.CV_HOUGH_MULTI_SCALE, 1, pi / 180, 50, 2, 2)
				for line in lines:
					cv.Line(src, line[0], line[1], cv.CV_RGB(255, 0, 0), 3, 8)
			cv.HoughCircles(smooth, circles, cv.CV_HOUGH_GRADIENT, 2, 100, 200, 100, 10, 500)
			for a in range(circles.rows):
				c_circle = circles[a, 0]
				c_radius = int(c_circle[2])
				c_x = int(c_circle[0])
				c_y = int(c_circle[1])
				cv.Circle(src, (c_x, c_y), c_radius, cv.RGB(255, 255, 255), 3, 8, 0)
		return lines, circles

	def capture_and_process(self):
		self.capturing = True
		p_img = self.img
		p_gray = cv.CreateImage(cv.GetSize(p_img), 8, 1)
		cv.CvtColor(p_img, p_gray, cv.CV_BGR2GRAY)
		while cv.GetCaptureProperty(self.capture, 1) < self.num_frames:
			cornersA = self.good_rect_features
			src_null = cv.QueryFrame(self.capture)
			if not(src_null):
				break
			#optimize scaling factor for screen resolt
			src = cv.CreateMat(self.img_height, self.img_width,cv.CV_8UC3)
			cv.Resize(src_null, src)
			self.num_frames-=1
			gray = cv.CreateImage(cv.GetSize(src), 8, 1)
			cv.CvtColor(src, gray, cv.CV_BGR2GRAY)
			lines, circles = self.get_lines_and_circles(src, gray)
			pyr_A = cv.CreateImage((p_gray.width + 8, gray.height/3),  cv.IPL_DEPTH_32F,1)
			pyr_B = cv.CreateImage((p_gray.width + 8, gray.height/3),  cv.IPL_DEPTH_32F,1)
			cornersB, features_found, features_errors = cv.CalcOpticalFlowPyrLK(p_gray,gray,pyr_A, pyr_B, cornersA,(5,5),5,(cv.CV_TERMCRIT_EPS, 20, 0.3), 0)
			p_gray = gray
			c_feature_displacement = []
			for feature in range(len(features_found)):
				if features_found[feature] and features_errors[feature]<=550:
					p0 = (cv.Round(cornersA[feature][0]), cv.Round(cornersA[feature][1]))
					p1 = (cv.Round(cornersB[feature][0]), cv.Round(cornersB[feature][1]))
					c_feature_displacement.append((p0,p1))
					cv.Circle(src,(int(p0[0]),int(p0[1])), 5, cv.RGB(17, 110, 255), 3, 8, 0)
				else: 
					c_feature_displacement.append((None,None))

			for rect in self.rect_features_all.keys():
				c_list = []
				print len(self.rect_features_all[rect]["features"]), self.rect_features_all[rect]["features"]
				for f in self.rect_features_all[rect]["features"]:
					print f
					c_list.append(c_feature_displacement[f])
				self.rect_features_all[rect]["displacement"].append(c_list)
			n_features = self.get_good_features_during_capture(src, p_gray)
			self.resolve_new_features(n_features, circles, lines)
			#need to resolve new features to previous shapes
			#for feature in n_features:
				#cv.Circle(src, (int(feature[0]), int(feature[1])), 5, cv.RGB(100, 100, 100), 3, 8, 0)
			cv.ShowImage("result", src)
			cv.WaitKey(30)
			#do analysis across the different features to pick up types of movement for a given rectangle(rotation, moving things back to previous location, splitting, turning vertical, 
			#write these data points to a rectangle specific data structure
			#aggregate movement and locations and then store for later processing. Also store data for each iteration in a csv or seperate files
			cornersA = cornersB
		#output=open("file_1","w")
		#print self.tracked_feature_displacement
		#output.write(str(self.tracked_feature_displacement))

	def resolve_new_features(self, features, circles, lines):
		for rect in self.rect_features_all.keys():
			sum = 0
			num_features = 0
			num_displacements = len(self.rect_features_all[rect]["displacement"])
			for displacement in self.rect_features_all[rect]["displacement"]:
				num_features = len(self.rect_features_all[rect]["displacement"][0])
				for x,y in displacement:
					if x and y:
						x_diff = x[0]-y[0]
						y_diff = x[1]-y[1]
						sum+= math.sqrt(math.pow(x_diff,2) + math.pow(y_diff,2))
			if num_displacements and num_features:
				if sum < 5:
					self.resolve_features_to_rectangles(rect, features, circles, lines)
					print "updated features"
				#need to restrict the resolving of new items to the rectangles that haven't been affected
		#look at the sum of the feature displacement. If it's small, then we can assume that the shape hasn't changed and then append the new circle or line assuming that it's the same

		#otherwise, for the features, if the feature locations are located inside or near the current circle, or between/near the lines that get returned

		#store the distance from the circle to the edge of the rectangle

	def detect_movement(self):
		for rect in self.rect_features_all.keys():
			for i in range(len(self.rect_features_all[rect]["displacement"]) -1):
			#find the center, length and width
			#for center take the average of all of the (x,y)'s that exist
			#for length take the (x,y) for the following (largest y, largest x; smallest y, smallest x) distance between these lines gives length (though this might be a problem when things are broken into parts). 
			# should also check for change in the distance between pairs of features within the same object
				if i>0:
					print "Incrementing"
					#looking at the dx,dy for respective pairs
					#detect occlusions
					#detect item removal from screen vs occlusions
		#also need to detect displacement of objects relative to one another and changes in these over time.
if __name__ == "__main__":
	app = QApplication(sys.argv)
	time = 0
	if len(sys.argv)>1:
		time = int(sys.argv[1])
	m = applicationWindow(frame_num=time)
	m.show()
	sys.exit(app.exec_())	

	