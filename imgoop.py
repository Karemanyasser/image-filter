import cv2
import numpy as np
from skimage.color import rgb2gray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Imagefourier:
    def __init__(self, image,mag_x,phase_x,y):
        self.image=cv2.imread(image,0)
        self.image=cv2.resize(self.image,(400,400))
        self.fftofimage=np.fft.fftshift(np.fft.fft2(self.image))
        self.magnitude=np.sqrt(np.real(self.fftofimage) ** 2 + np.imag(self.fftofimage) ** 2)
        self.magnitude_spectrum = 20*np.log(np.abs(self.fftofimage))
        self.phase=np.arctan2(np.imag(self.fftofimage), np.real(self.fftofimage))
        self.phase_spectrum = np.angle(self.fftofimage)
        self.mag_x=mag_x
        self.phase_x=phase_x
        self.mag_y=self.phase_y=y


class Processing(Imagefourier):

    def __init__(self,image,mag_x=0,phase_x=0,y=0):
        Imagefourier.__init__(self,image,mag_x,phase_x,y)
    
    def combine(self, magnitude, phase):
        self.outputcombine = np.multiply(magnitude, np.exp(1j *phase))
        self.img_output = np.real(np.fft.ifft2(self.outputcombine))

  
    def Mask_magnitude(self,x,y,width,height): 
        y_indx1= 300*(y-self.mag_y)/300
        y_indx2= 300*(y+height-self.mag_y)/300
        x_indx1= 300*(x-self.mag_x)/300
        x_indx2= 300*(x+width-self.mag_x)/300
        masked_mag=self.magnitude.copy()
        masked_mag[int(y_indx1):int(y_indx2),int(x_indx1):int(x_indx2)]=1
        print(int(y_indx1))
        print(y_indx2)
        print(x_indx1)
        print(x_indx2)
        return masked_mag

    def Mask_phase(self,x,y,width,height):
        y_indx1= 300*(y-self.phase_y)/300
        y_indx2= 300*(y+height-self.phase_y)/300
        x_indx1= 300*(x-self.phase_x)/300
        x_indx2= 300*(x+width-self.phase_x)/300
        masked_phase=self.phase.copy()
        masked_phase[int(y_indx1):int(y_indx2),int(x_indx1):int(x_indx2)]=1
        return masked_phase

    def reconstruct(self,image2,x,y,width,height,x2,y2,width2,height2):

        new_mag=np.array([])
        new_phase=np.array([])

        if ( all( self.mag_x+300>m> self.mag_x for m in (x,x+width)) and all( self.mag_y+300>m > self.mag_y for m in (y,y+height))):
            print("hena1")
            new_mag=self.Mask_magnitude(x,y,width,height)


        if ( all(image2.mag_x+300>m >image2.mag_x for m in (x,x+width)) and all(image2.mag_y+300>m >image2.mag_y for m in (y,y+height))):
            print("hena2")
            new_mag=image2.Mask_magnitude(x,y,width,height)

        if ( all( self.phase_x+300>m> self.phase_x for m in (x2,x2+width2)) and all( self.phase_y+300>m > self.phase_y for m in (y2,y2+height2))):
            print("hena3")
            new_phase=self.Mask_phase(x2,y2,width2,height2)



        if ( all( image2.phase_x+300>m> image2.phase_x for m in (x2,x2+width2)) and all( image2.phase_y+300>m > image2.phase_y for m in (y2,y2+height2))):
            print("hena4")
            new_phase=image2.Mask_phase(x2,y2,width2,height2)

        if (len(new_mag)!=0 and len(new_phase)!=0):
            self.combine(new_mag,new_phase)
            