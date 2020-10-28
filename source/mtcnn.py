import numpy as np
import mxnet as mx
import configparser
from core.symbol import P_Net, R_Net, O_Net
# from config import config
from core.detector import Detector
from core.fcn_detector import FcnDetector
from tools.load_model import load_param
from core.MtcnnDetector import MtcnnDetector

# --------------------------------------  load initial values ----------------------------------------


def load_mtcnn_params(config):
    mtcnn_config = config['MTCNN']
    mtcnn_prefix = eval(mtcnn_config['prefix'])

    mtcnn_epoch = eval(mtcnn_config['epoch'])
    mtcnn_batch_size = eval(mtcnn_config['batch_size'])
    mtcnn_thresh = eval(mtcnn_config['thresh'])
    mtcnn_min_face_size = int(mtcnn_config['min_face'])
    mtcnn_stride = mtcnn_config['stride']
    mtcnn_slide_window = False
    mtcnn_ctx = mx.cpu(0)

    detectors = [None, None, None]

    # load pnet model
    args, auxs = load_param(mtcnn_prefix[0], mtcnn_epoch[0], convert=True, ctx=mtcnn_ctx)
    if mtcnn_slide_window:
        PNet = Detector(P_Net("test"), 12, mtcnn_batch_size[0], mtcnn_ctx, args, auxs)
    else:
        PNet = FcnDetector(P_Net("test"), mtcnn_ctx, args, auxs)
    detectors[0] = PNet

    # load rnet model
    args, auxs = load_param(mtcnn_prefix[1], mtcnn_epoch[0], convert=True, ctx=mtcnn_ctx)
    RNet = Detector(R_Net("test"), 24, mtcnn_batch_size[1], mtcnn_ctx, args, auxs)
    detectors[1] = RNet

    # load onet model
    args, auxs = load_param(mtcnn_prefix[2], mtcnn_epoch[2], convert=True, ctx=mtcnn_ctx)
    ONet = Detector(O_Net("test"), 48, mtcnn_batch_size[2], mtcnn_ctx, args, auxs)
    detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, ctx=mtcnn_ctx, min_face_size=mtcnn_min_face_size,
                                   stride=mtcnn_stride, threshold=mtcnn_thresh, slide_window=mtcnn_slide_window)
    return mtcnn_detector

# -------------------------------------- end ----------------------------------------
