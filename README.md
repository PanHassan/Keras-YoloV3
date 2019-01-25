# Keras-YoloV3

inspired by qqwweee/keras-yolo3  https://github.com/qqwweee/keras-yolo3

Quick Start
Download YOLOv3 weights from YOLO website.
YoloV3 https://pjreddie.com/media/files/yolov3.weights

or

YoloV3-tiny https://pjreddie.com/media/files/yolov3-tiny.weights

Convert the Darknet YOLO model to a Keras model.
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5

Run YOLO detection.
wget https://pjreddie.com/media/files/yolov3.weights

Usage

python yolo_video.py --input test.mp4 --output out.mp4

Train
Generate your own annotation file and class names file.
One row for one image;
Row format: image_file_path box1 box2 ... boxN;
Box format: x_min,y_min,x_max,y_max,class_id (no space).

Example
path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3