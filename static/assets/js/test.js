const width = window.innerWidth;
const height = window.innerHeight;

var stage = new Konva.Stage({
  container: 'container',
  width: width,
  height: height,
});
var mag_selected=false;
var phase_selected=false;
var layer=new Konva.Layer();
var text = new Konva.Text({
    x: 1200,
    y: 10,
});
var text2 = new Konva.Text({
    x: 1200,
    y: 70,
});
layer.add(text2);
layer.add(text);
let  mag_rect = new Konva.Rect({});    
let phase_rect=new Konva.Rect({});
let isNowDrawing = false;
let mag_rect_is_drawn=false;
let phase_rect_is_drawn=false;
stage.add(layer);

//////////////FIRST IMAGE ,POSTING TO FLASK,RECEIVING IMAGE FFT:   
$("#file").change(function(e){
var URL = window.webkitURL || window.URL;
var url = URL.createObjectURL(e.target.files[0]);
var img = new Image();
img.src = url;
img.onload = function() {
var First_image = new Konva.Image({
    image: img,
    x: 50,
    y: 50,
    width: 300,
    height: 300,
});
First_image.cache();
layer.add(First_image);
}

input = $('#file')[0]
if(input.files && input.files[0])
{  
    let formData = new FormData();
    formData.append('image' , input.files[0]);
    formData.append('name' , 1);
    $.ajax({
        url: "/", 
        type:"POST",
        data: formData,
        cache: false,
        processData:false,
        contentType:false,
error: function(data){
            console.log("upload error" , data);
            console.log(data.getAllResponseHeaders());
        },
        success: function(data){
            bytestring = data['status']
            image = bytestring.split('\'')[1]
            bytestring2 = data['status2']
            image2 = bytestring2.split('\'')[1]
            // bytestring3 = data['status3']
            // image3 = bytestring3.split('\'')[1]

  var img1 = new Image();
  img1.onload = function () {
  var First_image_magnitude = new Konva.Image({
  x: 400,
  y: 50,
  image: img1,
  width: 300,
  height: 300,
});
layer.add(First_image_magnitude);

}; img1.src='data:image/jpeg;base64,'+image;

  var img2 = new Image();
  img2.onload = function () {
  var First_image_phase = new Konva.Image({
  x: 750,
  y: 50,
  image: img2,
  width: 300,
  height: 300,
});
layer.add(First_image_phase);

}; img2.src='data:image2/jpeg;base64,'+image2;

        }
    });
}
});

//////////////SECOND IMAGE UPLOAING,POSTING TO FLASK,RECEIVING IMAGE FFT:
$("#file1").change(function(e){
var URL = window.webkitURL || window.URL;
var url = URL.createObjectURL(e.target.files[0]);
var img = new Image();
img.src = url;
img.onload = function() {
var Second_image = new Konva.Image({
    image: img,
    x: 50,
    y: 410,
    width: 300,
    height:300,
});
Second_image.cache();
layer.add(Second_image);
}
input = $('#file1')[0]
if(input.files && input.files[0])
{  
    let formData = new FormData();
    formData.append('image' , input.files[0]);
formData.append('name' ,2);
$.ajax({
    url: "/", 
    type:"POST",
    data: formData,
    cache: false,
    processData:false,
    contentType:false,
error: function(data){
            console.log("upload error" , data);
            console.log(data.getAllResponseHeaders());
        },
        success: function(data){
            bytestring = data['status']
            image = bytestring.split('\'')[1]
            bytestring2 = data['status2']
            image2 = bytestring2.split('\'')[1]
            // bytestring3 = data['status3']
            // image3 = bytestring3.split('\'')[1]

  var img3 = new Image();
  img3.onload = function () {
  var second_image_magnitude = new Konva.Image({
  x: 400,
  y: 400,
  image: img3,
  width: 300,
  height: 300,
});
layer.add(second_image_magnitude);
}; img3.src='data:image/jpeg;base64,'+image;
 var img4 = new Image();
  img4.onload = function () {
  var second_image_phase = new Konva.Image({
  x: 750,
  y: 400,
  image: img4,
  width: 300,
  height: 300,
});
layer.add(second_image_phase);

}; img4.src='data:image2/jpeg;base64,'+image2;

// var img5 = new Image();
// img5.onload = function () {
// var reconstruct_image = new Konva.Image({
// x: 1065,
// y: 150,
// image: img5,
// width: 500,
// height:500,
// });
// layer.add(reconstruct_image);

// }; img5.src='data:image2/jpeg;base64,'+image3;
        }
    });
}
});

/////////// DRAWING RECTANGLE WITH MOUSE USING KONVA LIBRARY:
stage.on('mousedown ', function(){ 
    if (!mag_rect_is_drawn && mag_selected){
        
        isNowDrawing = true;
        mag_rect = new Konva.Rect({
        x: stage.getPointerPosition ().x,
        y: stage.getPointerPosition().y,
        width: 0,
        height: 0,
        fill: "lightblue",
        stroke: "blue",
        draggable: true,
        opacity: 0.5,

        });
        layer.add(mag_rect).batchDraw() ;
        }

    else if (!phase_rect_is_drawn && phase_selected){
           
        isNowDrawing = true;
        phase_rect = new Konva.Rect({
        x: stage.getPointerPosition ().x,
        y: stage.getPointerPosition().y,
        width: 0,
        height: 0,
        fill: "lightblue",
        stroke: "blue",
        draggable: true,
        opacity: 0.5,

        });
        layer.add(phase_rect).batchDraw() ;
        }


        }

     
);


stage.on ('mousemove ', function()
{if ( !isNowDrawing) return false;


    if (!mag_rect_is_drawn && mag_selected){

        var newWidth = stage.getPointerPosition().x - mag_rect.x();
        var newHeight = stage.getPointerPosition ().y - mag_rect.y();
        mag_rect.width(newWidth).height (newHeight);
        layer.batchDraw();
    
    }

    if (!phase_rect_is_drawn && phase_selected){
        var newWidth = stage.getPointerPosition().x - phase_rect.x();
        var newHeight = stage.getPointerPosition ().y - phase_rect.y();
        phase_rect.width(newWidth).height (newHeight);
        layer.batchDraw();
}
});

stage.on ('mouseup ', function(){             
if ( !isNowDrawing) return false;

isNowDrawing = false; 

if (mag_selected){
mag_rect_is_drawn=true;
var transforms1 = new Konva.Transformer({
nodes: [mag_rect],
});
transforms1.rotateEnabled(false);
layer.add(transforms1);

mag_rect.on('dragmove', function () {
  update_mag_Text();
});
mag_rect.on('transform', function () {
update_mag_Text();
console.log('transform');
});
}

if (phase_selected) { 
phase_rect_is_drawn=true;
var transforms2 = new Konva.Transformer({
nodes: [phase_rect],
}); 
transforms2.rotateEnabled(false);
layer.add(transforms2);
phase_rect.on('dragmove', function () {
  update_phase_Text();
});
phase_rect.on('transform', function () {
update_phase_Text();
console.log('transform');
});

}
});

document.getElementById('mag').addEventListener(
'click',
function () {
   phase_selected=false;
   mag_selected=true;
   mag_rect.moveToTop();
    var transforms3 = new Konva.Transformer({
nodes: [mag_rect],
}); 
transforms3.rotateEnabled(false);
layer.add(transforms3);
}
);

document.getElementById('phase').addEventListener(
'click',
function () {
    phase_selected=true;
    mag_selected=false;
    phase_rect.moveToTop();
    var transforms4 = new Konva.Transformer({
nodes: [phase_rect],


}); 
transforms4.rotateEnabled(false);
layer.add(transforms4);
}
);

function disabled(){
return disabled
}


function Reconstruct(){
    var coordinates = {
  'Mag_rectangle_x: ' : mag_rect.x() ,
    'Mag_rectangle_y: ' : mag_rect.y() ,
    'Mag_rectangle_width: ' : Math.abs(mag_rect.scaleX()*mag_rect.width()) ,
    'Mag_rectangle_height: ' : Math.abs(mag_rect.scaleY()* mag_rect.height()) ,
    'Phase_rectangle_x: ' :phase_rect.x() ,
    'Phase_rectangle_y: ' : phase_rect.y() ,
    'Phase_rectangle_width: ' : Math.abs(phase_rect.scaleX()*phase_rect.width()) ,
    'Phase_rectangle_height: ' : Math.abs(phase_rect.scaleY()* phase_rect.height()) }
;
//var coordinates = JSON.stringify(coordinates); 

$.ajax({
type: "POST",
url: "/",
contentType: "application/json",
data: JSON.stringify(coordinates),
processData: false,
error: function(data){
              console.log("upload error" , data);
              console.log(data.getAllResponseHeaders());
          },
          success: function(data){

    bytestring = data['status']
              image = bytestring.split('\'')[1]

    var img5 = new Image();
    img5.onload = function () {
    var reconstructed_img = new Konva.Image({
    x: 1065,
    y: 150,
    image: img5,
    width: 300,
    height: 300,
  });
  layer.add(reconstructed_img);
 
}; img5.src='data:image/jpeg;base64,'+image;
  }
});
  }

//////// UPDATING COORDINATES, WIDTH AND HEIGHT: 

function update_mag_Text() {
// var lines = [
// 'Mag_rectangle_x: ' + mag_rect.x() ,
// 'Mag_rectangle_y: ' + mag_rect.y() ,
// 'Mag_rectangle_width: ' + Math.abs(mag_rect.scaleX()*mag_rect.width()) ,
// 'Mag_rectangle_height: ' +Math.abs(mag_rect.scaleY()* mag_rect.height()) ,
// ];
// text.text(lines.join('\n'));

}  
function update_phase_Text() {
// var lines = [
// 'Phase_rectangle_x: ' + phase_rect.x(),
// 'Phase_rectangle_y: ' + phase_rect.y(),
// 'phase_rectangle_width: ' +Math.abs(phase_rect.scaleX()* phase_rect.width()),
// 'phase_rectangle_height: ' + Math.abs(phase_rect.scaleY()*phase_rect.height()),
// ];
// text2.text(lines.join('\n'));
}    
