import cv2
import configparser
from json import dumps
from base64 import b64encode
from websocket import create_connection
from mtcnn import load_mtcnn_params


class VideoProcess:
    def __init__(self, ws_conn, config):
        self.frame_skip = 2
        self.face_detector = load_mtcnn_params(config)
        self.ws_conn = ws_conn

    def serve_video(self, file):
        video = cv2.VideoCapture(file)
        frame_count = 0

        while True:
            _, image = video.read()
            if image is not None:
                if frame_count % self.frame_skip == 0:

                    # face detection
                    boxes, boxes_c = self.face_detector.detect_pnet(image)
                    boxes, boxes_c = self.face_detector.detect_rnet(image, boxes_c)
                    boxes, boxes_c = self.face_detector.detect_onet(image, boxes_c)

                    if boxes_c is not None:
                        for b in boxes_c:
                            cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)

                        # cv2.rectangle(image, (2800 - 100, 30), (3100, 150), (0, 0, 0), -1)
                        # cv2.putText(image, "count : " + str(len(boxes_c)), (2770, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        #                     1.5, (255, 255, 255), 3, cv2.LINE_AA)
                        # cv2.imwrite("frames/" + str(frame_count) + ".png", image)
                        _, buffer = cv2.imencode('.png', image)
                        b64_img = b64encode(buffer).decode('ascii')
                        self.ws_conn.send(dumps({"image": b64_img, "count": len(boxes_c)}))
                else:
                    pass
                frame_count += 1
            else:
                break


if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read('config.ini')
    ws_conn = config['SOCKET']['uri']
    video_file = config['VIDEO']['location']
    ws_conn = create_connection(ws_conn)
    ws_conn.send("hi")
    video_proc = VideoProcess(ws_conn, config)
    video_proc.serve_video(video_file)