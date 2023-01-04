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
from imgoop import Processing
import json

app = Flask(__name__,template_folder="templates")

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

def savefigures(magnitude_spectrum,mag_name,phase_spectrum,phase_name):
    fig =plt.figure(figsize=(15, 20))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.savefig(mag_name)
    plt.imshow(phase_spectrum, cmap='gray')
    plt.savefig(phase_name)      
    im1= Image.open(mag_name)
    im2= Image.open(phase_name) 
    data = io.BytesIO()
    im1.save(data, "JPEG")
    data2 = io.BytesIO()
    im2.save(data2, "JPEG")
    encoded_img_data1 = base64.b64encode(data.getvalue())
    encoded_img_data2=base64.b64encode(data2.getvalue())
    return encoded_img_data1,encoded_img_data2  


def save_output(img_output,name):
    fig =plt.figure(figsize=(15, 20))
    fig.patch.set_facecolor('#ffffff')
    desert_coffee_shift = img_output + img_output.min()
    desert_coffee_shift[desert_coffee_shift>255] = 255
    img_output[img_output>255] = 255
    img_output[img_output <0] = 0
    plt.imshow(img_output ,cmap='gray')
    fig =plt.figure(figsize=(15, 20))
    plt.imshow( img_output, cmap='gray')
    plt.savefig(name)
    im3= Image.open(name)
    data3 = io.BytesIO()
    im3.save(data3, "JPEG")
    encoded_img_data3=base64.b64encode(data3.getvalue())
    return encoded_img_data3   


image1_object=Processing('image1.jpg')
image2_object=Processing('image2.jpg')

@app.route('/', methods=['POST','GET'])
def home():

    global image1_object
    global image2_object

    if request.method == "POST":

        if  request.get_json() != None:
            output = request.get_json()
            co = json.loads(output)
            image1_object.reconstruct(image2_object,co['Mag_rectangle_x: ' ]
            ,co['Mag_rectangle_y: '],co['Mag_rectangle_width: ' ],co['Mag_rectangle_height: '],
            co['Phase_rectangle_x: '],co['Phase_rectangle_y: '],co['Phase_rectangle_width: '],co[ 'Phase_rectangle_height: ' ])
            print(co)
            rec=save_output(image1_object.img_output,"reconstructed.jpg")
            return jsonify({'status':str(rec)})

        
        file= request.files['image'].read()   
        default_value = '0'
        name = request.form.get('name', default_value)   
        write_file_to_image(file)  
        img = skimage.io.imread("image.jpg")

       
        
        if name==str(1):
            cv2.imwrite('image1.jpg', img)  
            image1_object =Processing('image1.jpg',400,750,60)
            encoded_img_data1,encoded_img_data2= savefigures(image1_object.magnitude_spectrum,"mag1.jpg",image1_object.phase_spectrum,"phase1.jpg")
            return jsonify({'status':str(encoded_img_data1),'status2':str(encoded_img_data2)})

        elif name==str(2):
            cv2.imwrite('image2.jpg', img)
            image2_object=Processing('image2.jpg',400,750,400)
            encoded_img_data1,encoded_img_data2=savefigures(image2_object.magnitude_spectrum,"mag2.jpg",image2_object.phase_spectrum,"phase2.jpg")
            return jsonify({'status':str(encoded_img_data1),'status2':str(encoded_img_data2)})

        
        
    return render_template('test.html')


if __name__ == '__main__':
   app.run(debug=True)