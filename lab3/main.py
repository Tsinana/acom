import cv2 as cv
from classes.detecting_motion_in_video import DetectingMotionInVideo

from classes.my_kanni import MyKanni



if __name__ == '__main__':
    # MyKanni.start('other\\text.jpg',(3,3),0.1,0.9,1)
    DetectingMotionInVideo.start((3,3),0.1)
