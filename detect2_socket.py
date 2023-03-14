# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import torch
# added by RHP
from time import time
import asyncio
import threading
import math
import distance
# web streaming library(RHP)
from flask import Flask, Response, render_template
from threading import Thread
from queue import Queue
import cx_Oracle

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import socket

# declaration by RHP
# mkBool : 생성해도 되는지 여부를 나타내는 Boolean 변수(True : 가능, False : 불가)
def mkTxt(txt_path, line, mkBool) :
    if(mkBool[0] == True):
        with open(f'{txt_path}.txt', 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

# declaration by RHP
# 1초마다 txt 파일을 생성 가능하게 해주는 역할
def alterMkBool(mkBool):
    mkBool[0] = True
    threading.Timer(1, alterMkBool, [mkBool]).start()

# made by RHP
mkBool = [False]

# 스레드를 시작해주는 부분
t = threading.Thread(target=alterMkBool, args=(mkBool,))
t.start()

app = Flask(__name__)



# 서버의 주소와 포트번호를 지정합니다.(NHJ)
HOST = '127.0.0.1'
PORT = 9999

# 소켓 객체를 생성합니다.
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 서버의 주소와 포트번호를 바인딩합니다.
server_socket.bind((HOST, PORT))
# 클라이언트의 연결 요청을 대기합니다.
server_socket.listen()

# 클라이언트와 연결이 되면, detection 결과를 수신하는 함수입니다.
def receive_result(conn):
    # 결과의 길이를 수신합니다.
    data = conn.recv(16)
    result_len = int(data.decode())
    # 결과를 수신합니다.
    data = conn.recv(result_len)
    return data.decode()

while True:
    # 클라이언트의 연결 요청이 있을 때까지 대기합니다.
    conn, addr = server_socket.accept()
    # detection 결과를 수신합니다.
    result = receive_result(conn)
    # 결과를 처리합니다.
    print(result)
    # 소켓 연결을 종료합니다.
    conn.close()

@app.route("/") 
def home(): 
    return render_template('index.html')

@app.route("/getimg")
def getimg():
    return Response(generate_cam00(), mimetype = "multipart/x-mixed-replace; boundary=frame")



# 소켓 객체를 생성합니다.
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 서버에 연결합니다.
client_socket.connect((HOST, PORT))

# detection 결과를 소켓을 통해 클라이언트에게 전송하는 함수입니다.
def send_result(results):
    # results를 bytes로 인코딩합니다.
    encoded_results = str(results).encode()
    # 결과의 길이를 전송합니다.
    client_socket.send(str(len(encoded_results)).ljust(16).encode())
    # 결과를 전송합니다.
    client_socket.send(encoded_results)

def generate_cam00():
    weights='./best.pt'  # model path or triton URL
    source=str('0') #ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
    data='C:/Users/smhrd/yolov5/data.yaml' #ROOT / 'data/coco128.yaml',  # dataset.yaml path
    imgsz=(640, 640)  # inference size (height, width)
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    device='cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=True  # show results
    save_img=True
    save_txt=True  # save results to *.txt
    save_conf=False  # save confidences in --save-txt labels
    save_crop=False  # save cropped prediction boxes
    nosave=False  # do not save images/videos
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    visualize=False  # visualize features
    update=False  # update all models
    project=ROOT / 'runs/detect'  # save results to project/name
    name='exp'  # save results to project/name
    exist_ok=True  # existing project/name ok, do not increment
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=True  # hide confidences
    half=False  # use FP16 half-precision inference
    dnn=False  # use OpenCV DNN for ONNX inference
    vid_stride=1  # video frame-rate stride

    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric()# or source.endswith('.streams') or (is_url)# and not is_file)
    # screenshot = source.lower().startswith('screen')

    # if is_url: # and is_file:
        # source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    label_dir_path = "./runs/detect/exp/labels/"
    # 이전에 있었던 라벨 파일들 삭제
    for f in os.listdir(label_dir_path):
        os.remove(os.path.join(label_dir_path, f))
    print('라벨 데이터 초기화 완료')

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
        print("웹캠 실행")
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    for path, im, im0s, vid_cap, s in dataset:

        start_time = time()

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            new_frame = format(frame, '10')
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{new_frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        mkTxt(txt_path, line, mkBool)

                        # txt 파일을 만든 후 거리 계산
                        distance.calDist()
                        
                        # functionalized by RHP
                        # with open(f'{txt_path}.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # added by RHP
                # 1초마다 True로 변환 시킬 것임.
                mkBool[0] = False

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                #cv2.imshow(str(p), im0) # str(p) : 표시될 창 이름, im0 : 이미지
                #cv2.waitKey(1)  # 1 millisecond

                # encode the frame in JPEG format(RHP)
                flag, encodedImage = cv2.imencode(".jpg",im0)
                # socket으로 결과전송
                send_result(im0.xyxy)

                # yield the output frame in the byte format(RHP)
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')




            # Save results (image with detections)
            if save_img:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

        # Print time (inference-only)
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        end_time = time()
        fps = 1/(end_time-start_time)
        print("fps : ", fps)

    # Print results
    # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)



# def main(opt): 
#     check_requirements(exclude=('tensorboard', 'thop'))
#     run(**vars(opt))
    

if __name__ == '__main__':
    # opt = parse_opt()
    # main(opt)

    # start the flask app
    # 127.0.0.1:5500/getimg
    app.run(host='127.0.0.1', port=5500, debug=True, threaded=True, use_reloader=False)

    
generate_cam00()