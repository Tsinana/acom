import cv2 as cv
import numpy as np
from classes.my_gaussian_blur import MyGaussianBlur
from classes.print_img import PrintImg
from classes.my_kanni import MyKanni


class DetectingMotionInVideo:
  @classmethod
  def start(cls, filter_shape, sigma):
    cap = cv.VideoCapture("C:\\Users\\Tsinana\\GitHub\\7-th_semester\\ACOM\\lab3\\main_video.mov")
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('out.mov', fourcc, 60.0, (640, 480))
    saved_frame = None
    flag = False
    save_flag = False
    try:
      while cap.isOpened():
        ret, new_frame = cap.read()
        if not ret:
          print("Can't receive frame (stream end?). Exiting ...")
          break
        if save_flag:
          saved_frame = new_frame

        new_frame = cls.prepare_imgframe(new_frame,filter_shape, sigma)
        if flag:
          frame_being_processed = cv.absdiff(new_frame,old_frame)
          _,threshed_frame = cv.threshold(frame_being_processed, 33, 255, cv.THRESH_BINARY)

          threshed_frame = np.array(threshed_frame, np.uint8)
          contours,_ = cv.findContours(threshed_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
          for contour in contours:
            if(cv.contourArea(contour) > 200):
              if save_flag:
                out.write(saved_frame)
              save_flag = True
            else:
              save_flag = False


        old_frame = new_frame
        flag = True


        if cv.waitKey(1) == ord('q'):
          break
    finally:
      cap.release()
      out.release()
      cv.destroyAllWindows()

  @classmethod
  def prepare_imgframe(cls,img, filter_shape, sigma):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return cv.GaussianBlur(gray, (15, 15), 1)
