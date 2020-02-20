from ctypes import *
import math
import random

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("../libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res


####################### WEB #############################


import os
from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory
from werkzeug.utils import secure_filename
import time
import cv2

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

app = Flask(__name__, static_url_path = '/static/')
app.secret_key = "super secret key"

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

objects = ['ManyObjects', 'Monkey', 'All']
select = 0
results = []
image_name = ""
image_predict = ""


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

	
@app.route('/')
def index():
	return render_template('upload.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    os.system('rm static/*.jpg')
    os.system('rm static/*.jpeg')
    global image_name
    global image_predict
    if request.method == 'POST':
        target = os.path.join(APP_ROOT, 'static')
        if not os.path.isdir(target):
            os.mkdir(target)
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            image_name = filename
            image_predict = 'predict_' + filename 
            destination = "/".join([target, filename])
            file.save(destination)
            return render_template('detector.html', objects=objects, image_name=filename)
    return render_template('upload.html')


@app.route('/detector', methods=['GET', 'POST'])
def detector():
    global results
    results = []
    r = []
    if request.method == 'POST':
        image_file_path = 'static/' + image_name
        if request.form.get('select_objects') == 'Monkey':
            r = detect(net_yolov3_monkey, meta_yolov3_monkey, image_file_path)
            boundingBox(image_file_path, r)
        else:
            if request.form.get('select_objects') == 'ManyObjects':
                r = detect(net_yolov3, meta_yolov3, image_file_path)
                boundingBox(image_file_path, r)
            else:
                r1 = detect(net_yolov3_monkey, meta_yolov3_monkey, image_file_path)
                r2 = detect(net_yolov3, meta_yolov3, image_file_path)
                r= r1 + r2
                boundingBox(image_file_path, r)

        for res in r:
            accuracy = str(float(res[1])*100)
            results.append([res[0], accuracy])
        if len(results) == 0:
            results.append(["No Objects","None"])
        
        return render_template('detector_completed.html', results = results, redicted = image_predict)

    return render_template('upload.html')

#fontFace = cv2.FONT_ITALIC
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (0, 0, 0)

MAIN_COLORS = [(0,255,0),(255,0,0),(0,255,255),(255,255,0),(0,0,255),(255,0,255),(192,192,192),(128,128,128),(0,0,128),(0,128,128),(0,128,0),(128,0,128),(128,128,0),(128,0,0)]

def getColorBox(r):
    colorObjects = {}
    index_color = 0
    for i in range(0,len(r)):
        if r[i][0] not in colorObjects.keys():
            colorObjects.update({r[i][0]: MAIN_COLORS[index_color]})
            index_color = index_color + 1
    return colorObjects

def boundingBox(fileName, r):
    colorObjects = getColorBox(r)
    des = 'static/' + image_predict
    img = cv2.imread(fileName)
    for i in range(len(r)):
        lable = r[i][0]
        accuracy = float(r[i][1])*100
        lable = lable + ': ' + '{:^2.0f}'.format(accuracy) + '%'
        x = int(r[i][2][0])
        y = int(r[i][2][1])
        w = int(r[i][2][2])
        h = int(r[i][2][3])
        cv2.rectangle(img, (x-w/2,y-h/2), (x+w/2,y+h/2), colorObjects[r[i][0]], 2)
        cv2.rectangle(img, (x-w/2,y-h/2), (x-w/2+10*len(lable),y-h/2+15),colorObjects[r[i][0]], cv2.FILLED)
        cv2.putText(img, lable, (x-w/2+2,y-h/2+12), fontFace, fontScale, fontColor)
    cv2.imwrite(des, img)

@app.route('/completed')
def completed():
    if request.method == 'POST':
        return render_template('detector_completed.html')
    else:
        return render_template('upload.html')
########################### END WEB ##################################

net_yolov3 = load_net("../cfg/yolov3.cfg", "../yolov3.weights", 0)
meta_yolov3 = load_meta("../cfg/coco.data")
net_yolov3_monkey = load_net("../cfg/monkey_yolov3.cfg", "../monkey_yolov3_new_21000.weights", 0)
meta_yolov3_monkey = load_meta("../cfg/monkey.data")

if __name__ == "__main__":
    app.run(debug = True)
