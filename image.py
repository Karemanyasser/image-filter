import cv2
import numpy as np
from skimage.color import rgb2gray

class Image:
    def __init__(self, name , image,mag_x=0,phase_x=0,y=0):
        self.name = name
        self.image=cv2.imread(image,0)
        self.image=cv2.resize(self.image,(400,400)) 
        self.fftofimage=np.fft.fftshift(np.fft.fft2(self.image))
        self.magnitude=np.sqrt(np.real(self.fftofimage) ** 2 + np.imag(self.fftofimage) ** 2)
        self.phase=np.arctan2(np.imag(self.fftofimage), np.real(self.fftofimage))
        self.mag_x=mag_x
        self.phase_x=phase_x
        self.mag_y=self.phase_y=y

  
    def Mask_magnitude(self,x,y,width,height): 
        self.y_indx1= 300*(y-self.mag_y)/300
        self.y_indx2= 300*(y+height-self.mag_y)/300
        self.x_indx1= 300*(x-self.mag_x)/300
        self.x_indx2= 300*(x+width-self.mag_x)/300
        self.masked_mag=self.magnitude.copy()
        self.masked_mag[int(self.y_indx1):int(self.y_indx2),int(self.x_indx1):int(self.x_indx2)]=1
    
    def Mask_phase(self,x,y,width,height):
            self.y_indx1= 300*(y-self.phase_y)/300
            self.y_indx2= 300*(y+height-self.phase_y)/300
            self.x_indx1= 300*(x-self.phase_x)/300
            self.x_indx2= 300*(x+width-self.phase_x)/300
            self.masked_phase=self.phase.copy()
            self.masked_phase[int(self.y_indx1):int(self.y_indx2),int(self.x_indx1):int(self.x_indx2)]=1

    
            

  
