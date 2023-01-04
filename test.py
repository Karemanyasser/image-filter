from flask import Flask, render_template, request , jsonify
import numpy as np
from scipy.fft import fft2 ,ifft2
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import io 
import base64 
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import plotly.graph_objs as go
import plotly.express as px
import skimage.io 
from skimage.color import rgb2gray
import json

app = Flask(__name__,template_folder="templates")

def fourier(img,width,height,mag_name,phase_name):
    file=cv2.imread(img,0)
    file=cv2.resize(file, (width,height))
    file_fft = np.fft.fftshift(np.fft.fft2(file))
    magnitude = np.sqrt(np.real(file_fft) ** 2 + np.imag(file_fft) ** 2)
    magnitude_spectrum = 20*np.log(np.abs(file_fft))
    phase = np.arctan2(np.imag(file_fft), np.real(file_fft))
    phase_spectrum = np.angle(file_fft)
    fig =plt.figure(figsize=(15, 20))
    plt.imshow( magnitude_spectrum,cmap='gray')
    plt.savefig(mag_name)
    plt.imshow(phase_spectrum,cmap='gray')
    plt.savefig(phase_name)
    return magnitude, phase

def inverse(magnitude,phase,name):
    output = np.multiply(magnitude, np.exp(1j *phase))
    img_output = np.real(np.fft.ifft2(output))  
    fig =plt.figure(figsize=(15, 20))
    fig.patch.set_facecolor('#ffffff')
    desert_coffee_shift = img_output + img_output.min()
    desert_coffee_shift[desert_coffee_shift>255] = 255
    img_output[img_output>255] = 255
    img_output[img_output <0] = 0
    plt.imshow(img_output )
    fig =plt.figure(figsize=(15, 20))
    plt.imshow( img_output,cmap='gray')
    plt.savefig(name)

def send_plts(file_name1,file_name2,):
    im1= Image.open(file_name1)
    im2= Image.open(file_name2) 
    data = io.BytesIO()
    im1.save(data, "JPEG")
    data2 = io.BytesIO()
    im2.save(data2, "JPEG")
    encoded_img_data1 = base64.b64encode(data.getvalue())
    encoded_img_data2=base64.b64encode(data2.getvalue())
    return encoded_img_data1,encoded_img_data2

def send(file_name3):
    im3= Image.open(file_name3)
    data3 = io.BytesIO()
    im3.save(data3, "JPEG")
    encoded_img_data3=base64.b64encode(data3.getvalue())
    return encoded_img_data3

def write_file_to_image(file):
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    with open("image.jpg", "wb") as fh:
        fh.write(base64.decodebytes(img_base64)) 


def Mask_magnitude(mag_x,mag_y,x,y,width,height,mag): 
    y_indx1= 300*(y-mag_y)/300
    y_indx2= 300*(y+height-mag_y)/300
    x_indx1= 300*(x-mag_x)/300
    x_indx2= 300*(x+width-mag_x)/300
    masked_mag=mag.copy()
    masked_mag[int(y_indx1):int(y_indx2),int(x_indx1):int(x_indx2)]=1
    print(int(y_indx1))
    print(y_indx2)
    print(x_indx1)
    print(x_indx2)
    return masked_mag

def Mask_phase(phase_x,phase_y,x,y,width,height,phase):
    y_indx1= 300*(y-phase_y)/300
    y_indx2= 300*(y+height-phase_y)/300
    x_indx1= 300*(x-phase_x)/300
    x_indx2= 300*(x+width-phase_x)/300
    masked_phase=phase.copy()
    masked_phase[int(y_indx1):int(y_indx2),int(x_indx1):int(x_indx2)]=1
    return masked_phase

def reconstruct(mag,phase,x,y,width,height,x2,y2,width2,height2,mag_x,mag_y,phase_x,phase_y):
    new_mag=np.array([])
    new_phase=np.array([])

    if ( all( mag_x+300>m> mag_x for m in (x,x+width)) and all( mag_y+300>m > mag_y for m in (y,y+height))):
        print("hena1")
        new_mag=Mask_magnitude(mag_x,mag_y,x,y,width,height,mag)

    if ( all(mag_x+300>m >mag_x for m in (x,x+width)) and all(mag_y+300>m >mag_y for m in (y,y+height))):
        print("hena2")
        new_mag=Mask_magnitude(mag_x,mag_y,x,y,width,height,mag)

    if ( all( phase_x+300>m> phase_x for m in (x2,x2+width2)) and all( phase_y+300>m > phase_y for m in (y2,y2+height2))):
        print("hena3")
        new_phase=Mask_phase(phase_x,phase_y,x2,y2,width2,height2,phase)

    if ( all( phase_x+300>m> phase_x for m in (x2,x2+width2)) and all( phase_y+300>m > phase_y for m in (y2,y2+height2))):
        print("hena4")
        new_phase=Mask_phase(phase_x,phase_y,x2,y2,width2,height2,phase)

    if (len(new_mag)!=0 and len(new_phase)!=0):
        fig = plt.figure(figsize=(4, 4))
        complex_f=new_mag* np.exp(1j* new_phase)
        reconstructed=np.fft.ifft2(complex_f)
        plt.imshow(np.abs(reconstructed), cmap='gray')
        plt.savefig('reconstructed.jpg')  

data=[]
images=[]

@app.route('/', methods=['POST','GET'])
def home():

    global mag_img1
    global phase_img1
    global mag_img2
    global phase_img2

    if request.method == "POST":
        if  request.get_json() != None:
            output = request.get_json()
            co = json.loads(output)
            reconstruct(mag_img2,phase_img2,co['Mag_rectangle_x: ' ]
            ,co['Mag_rectangle_y: '],co['Mag_rectangle_width: ' ],co['Mag_rectangle_height: '],
            co['Phase_rectangle_x: '],co['Phase_rectangle_y: '],co['Phase_rectangle_width: '],co[ 'Phase_rectangle_height: ' ],400,400,750,400)
            print(co)
            rec=send("reconstructed.jpg")
            return jsonify({'status':str(rec)})
        file= request.files['image'].read()   
        default_value = '0'
        name = request.form.get('name', default_value)   
        write_file_to_image(file)  
        img = skimage.io.imread("image.jpg")
        
        if name==str(1):
            cv2.imwrite('image1.jpg', img)  
            original_image = cv2.imread('image1.jpg')
            mag_img1, phase_img1 =fourier('image1.jpg',400,400,"mag1.jpg","phase1.jpg")
            data.append(mag_img1)
            data.append(phase_img1)
            encoded_img_data1,encoded_img_data2=send_plts('mag1.jpg','phase1.jpg')
            inverse(mag_img1,phase_img1,"rec.jpg")
            rec=send("rec.jpg")
            return jsonify({'status':str(encoded_img_data1),'status2':str(encoded_img_data2)})
        elif name==str(2):
            cv2.imwrite('image2.jpg', img)
            original_image = cv2.imread('image2.jpg')
            mag_img2, phase_img2=fourier('image2.jpg',400,400,"mag2.jpg","phase2.jpg")
            encoded_img_data1,encoded_img_data2=send_plts('mag2.jpg','phase2.jpg')
            data.append(mag_img2)
            data.append(phase_img2)
            inverse(data[0],data[3],"rec1.jpg")
            rec=send("rec1.jpg")
            data.clear()
            return jsonify({'status':str(encoded_img_data1),'status2':str(encoded_img_data2)})
        
        

    return render_template('test.html')


if __name__ == '__main__':
   app.run(debug=True)