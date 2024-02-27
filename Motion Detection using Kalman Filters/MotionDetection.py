import sys
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import dilation
from PySide2 import QtCore, QtWidgets, QtGui
from skvideo.io import vread

from PIL.ImageQt import ImageQt
from PIL import Image

list1=[]
class MotionDetector:
    def __init__(self,frames):
        
        self.frame_idx = 3  #Frame hysteresis for determining active or inactive objects.
        self.threshold = 0.05 #The motion threshold for filtering out noise.
        self.dist_thresh = 1 #Distance threshold to determine if an object candidate belongs to an object currently being tracked.
        self.frames_to_skip = 60  # s - The number of frames to skip between detections
        self.max_obj = 6  # N - The number of maximum objects to detect.
        
        self.KF = KalmanFilter()
        
        self.ppframe = rgb2gray(frames[self.frame_idx-2])
        self.pframe = rgb2gray(frames[self.frame_idx-1])
        self.cframe = rgb2gray(frames[self.frame_idx])
        self.cframe_rgb = (frames[self.frame_idx])
        self.diff1 = np.abs(self.cframe - self.pframe)
        self.diff2 = np.abs(self.pframe - self.ppframe)
        
        self.motion_frame = np.minimum(self.diff1, self.diff2)
        self.thresh_frame = self.motion_frame > self.threshold
        self.dilated_frame = dilation(self.thresh_frame, np.ones((9, 9)))
        self.label_frame = label(self.dilated_frame)
        self.regions = regionprops(self.label_frame)
    
    def update_frames(self, isIncreased):
        if isIncreased == True:
            self.frame_idx += 10
        else:
            self.frame_idx -= 10
        
        self.ppframe = rgb2gray(frames[self.frame_idx-2])
        self.pframe = rgb2gray(frames[self.frame_idx-1])
        self.cframe = rgb2gray(frames[self.frame_idx])
        self.cframe_rgb = (frames[self.frame_idx])
        self.diff1 = np.abs(self.cframe - self.pframe)
        self.diff2 = np.abs(self.pframe - self.ppframe)
        
        self.motion_frame = np.minimum(self.diff1, self.diff2)
        self.thresh_frame = self.motion_frame > self.threshold
        self.dilated_frame = dilation(self.thresh_frame, np.ones((9, 9)))
        self.label_frame = label(self.dilated_frame)
        self.regions = regionprops(self.label_frame)
        
    def get_regions(self):
        return self.regions
    
    def get_cframe(self):
        return self.cframe_rgb
    
    def get_frame_idx(self):
        return self.frame_idx
    
class KalmanFilter:
    def __init__(self):
        self.state = 1
        self.P = 1
        self.noise = 1
        self.measFunc = np.matrix([[1,0,0,0], [0,0,1,0]])
    
    def predict(self):
        self.predState = np.round(np.dot(self.noise, self.state))
        self.predP = np.dot(self.predState, np.dot(self.P, self.state.T))
        + self.noise
        
        return self.predP
    
    def update(self, y):  
        temp = np.linalg.pinv(self.measFunc*self.predP*self.measFunc.T)
        self.kalmanGain = self.predP*self.measFunc.T*temp
        
        self.state = self.kalmanGain*(self.measFunc-(self.noise*self.predP))
        self.state += self.predState
        
        self.P = self.kalmanGain * self.state * self.noise
        
    
class QtDemo(QtWidgets.QWidget):
    def __init__(self, frames):
        super().__init__()

        self.frames = frames

        self.current_frame = 0

        self.button_next = QtWidgets.QPushButton("Next Frame")
        self.button_back = QtWidgets.QPushButton("Back Frame")
        
        #create Motion Detection Object
        self.motion = MotionDetector(self.frames)   

        # Configure image label
        self.img_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        h, w, c = self.frames[0].shape
        if c == 1:
            img = QtGui.QImage(self.frames[0], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[0], w, h, QtGui.QImage.Format_RGB888)
        
        
        img_plot = self.create_plot(False)
        img_plot.savefig('frame_img.png') 
        img = QtGui.QImage("frame_img.png")
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))


        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.img_label)
        self.layout.addWidget(self.button_next)
        self.layout.addWidget(self.button_back)

        # Connect functions
        self.button_next.clicked.connect(self.on_click_next)
        self.button_back.clicked.connect(self.on_click_back)
        

    @QtCore.Slot()
    def on_click_next(self):
        if self.current_frame == self.frames.shape[0]-1:
            return
        for i in range(6):
            self.motion.update_frames(True) # true means increase frames
            self.img_plot = self.create_plot(True)
            self.img_plot.savefig('frame_img.png') 
            self.img = QtGui.QImage("frame_img.png")
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(self.img))
        self.current_frame += 60
        
    def on_click_back(self):
        if self.current_frame == self.frames.shape[0]-1:
            return
        
        for i in range(6):
            self.motion.update_frames(False) # false means decrease frames
        list1.clear()
        self.img_plot = self.create_plot(False)
        self.img_plot.savefig('frame_img_back.png') 
        self.img = QtGui.QImage("frame_img_back.png")
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(self.img))
        self.current_frame -= 60
        

    def draw_bbox(self,ax, bbox, isNext):
        
        minr, minc, maxr, maxc = bbox
        bx = [minc, maxc, maxc, minc, minc]
        by = [minr, minr, maxr, maxr, minr]
        ax.plot(bx, by, '-b', linewidth=2)
        if(isNext):
            list2=[0,0,0,0]
            list2[0]=minr
            list2[1]=minc
            list2[2]=maxr
            list2[3]=maxc
            list1.append(list2)
            for i in range(len(list1)):
                circle1 = plt.Circle((int((list1[i][3]-list1[i][1])//2)+list1[i][1],int((list1[i][2]-list1[i][0])//2)+list1[i][0]), 1, color='r')
                ax.add_patch(circle1)
            
    
    def create_plot(self, isNext):
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cframe_rgb = self.motion.get_cframe()
        ax.imshow(cframe_rgb)
        ax.set_axis_off()
        ax.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
        regions = self.motion.get_regions()
        for r in regions:
            self.draw_bbox(ax, r.bbox, isNext)
        plt.close(fig)            
        
        return fig

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Demo for loading video with Qt5.")
    parser.add_argument("video_path", metavar='PATH_TO_VIDEO', type=str)
    parser.add_argument("--num_frames", metavar='n', type=int, default=-1)
    parser.add_argument("--grey", metavar='True/False', type=str, default=False)
    args = parser.parse_args()

    num_frames = args.num_frames

    if num_frames > 0:
        frames = vread(args.video_path, num_frames=num_frames, as_grey=args.grey)
    else:
        frames = vread(args.video_path, as_grey=args.grey)

    app = QtWidgets.QApplication([])

    widget = QtDemo(frames)
    widget.resize(600, 400)
    widget.show()

    sys.exit(app.exec_())