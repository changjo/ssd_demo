from __future__ import print_function
import cv2
import time
import numpy as np
#from imutils.video import FPS, WebcamVideoStream
import argparse
import sys
from os import path
sys.path.append(path.dirname(sys.path[0]))

from utils import save_images

import pickle

from concurrent import futures
import grpc
import exchange_frame_pb2
import exchange_frame_pb2_grpc

from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--load_video', default=None, type=str,
                    help='Load video path')
parser.add_argument('--save_video', default=None, type=str,
                    help='Save video path')
parser.add_argument('--confidence_threshold', default=0.5, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--grpc_server_ip', type=str, required=True,
                    help='gRPC server ip address (ex: 127.0.0.1:50051')

args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def multithreading(func, args, workers):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        res = executor.map(func, args)
    return list(res)


def run(): 
    def call_grpc(cap, values_deque, threading_lock):
        threading_lock.acquire()
        ret, frame = cap.read()
        threading_lock.release()
        input_frame = cv2.resize(frame, (300, 300))
        
        frame_bytes = pickle.dumps(input_frame)
        if 1: 
            response = stub.FrameBoxesBytesProcess(exchange_frame_pb2.InputFrame(frame=frame_bytes))
        if 0: # async
            response_future = stub.FrameBoxesBytesProcess.future(exchange_frame_pb2.InputFrame(frame=frame_bytes))
            response = response_future.result()
        
        boxes, labels, confs = pickle.loads(response.boxes)
        frame = save_images.draw_images(frame, np.hstack([boxes, labels]), confs, color_bgr2rgb=False)

        values_deque.append(frame)
        #return response, frame
    
    channel = grpc.insecure_channel(args.grpc_server_ip)
    stub = exchange_frame_pb2_grpc.FrameGetterStub(channel)
    
    print("[INFO] starting threaded video stream...")
    if args.load_video is None:
        cap = cv2.VideoCapture(0)
        cap.set(3, 1920)
        cap.set(4, 1080)
    else:
        cap = cv2.VideoCapture(args.load_video)
    width, height = int(cap.get(3)), int(cap.get(4))
    print("[i] Resolution: {} x {}".format(width, height))
    #print("[i] Confidence Threshold: {}".format(args.confidence_threshold))

    if args.save_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.save_video, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
    
    # Initial dummy grpc request to avoid "CUDA error: out of memory"
    stub.FrameBoxesBytesProcess(exchange_frame_pb2.InputFrame(frame=pickle.dumps(np.zeros((3, 3, 3)))))
    time.sleep(1.0)
    
    threading_lock = threading.Lock()  

    num_frames = 30
    k = 0
    values_deque = deque()
    threads = deque(maxlen=1000)
    while cap.isOpened():
        key = cv2.waitKey(1) & 0xFF
        if k == 0: 
            start_time = time.time()
        
        temp_start_time = time.time()
        if threading.active_count() < 4:
            proc = threading.Thread(target=call_grpc, args=(cap, values_deque, threading_lock))
            proc.start()
            threads.append(proc)
            #proc.join() 
            if len(values_deque) > 0:
                frame = values_deque.popleft()
                # keybindings for display
                if key == ord('p'):  # pause
                    while True:
                        key2 = cv2.waitKey(1) or 0xff
                        cv2.imshow('frame', frame)
                        if key2 == ord('p'):  # resume
                            break
                
                cv2.imshow('frame', frame)
                             
                if args.save_video is not None:
                    out.write(frame)

                k += 1
                if k == num_frames:
                    seconds = time.time() - start_time
                    sys.stdout.write("\r{:.2f} fps".format(num_frames / seconds))
                    sys.stdout.flush()
                    k = 0

        if key == 27:  # exit
            break
    
    for proc in threads:
        proc.join()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    
    run()
