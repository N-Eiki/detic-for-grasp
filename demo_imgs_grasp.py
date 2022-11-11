# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import copy
import sys
import mss

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo

# Fake a video capture object OpenCV style - half width, half height of first screen using MSS
class ScreenGrab:
    def __init__(self):
        self.sct = mss.mss()
        m0 = self.sct.monitors[0]
        self.monitor = {'top': 0, 'left': 0, 'width': m0['width'] / 2, 'height': m0['height'] / 2}

    def read(self):
        img =  np.array(self.sct.grab(self.monitor))
        nf = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return (True, nf)

    def isOpened(self):
        return True
    def release(self):
        return True


# constants
WINDOW_NAME = "Detic"

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, args, outline=True)

    files = os.listdir(args.input)
    img_paths = [os.path.join(args.input, file) for file in files]
    if args.input:
        for path in tqdm.tqdm(img_paths): 
        # for path in tqdm.tqdm(args.input, disable=not args.output):
            img = read_image(path, format="BGR")
            img = copy.copy(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)

            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            out_path = os.path.join(args.output, path.split("/")[-1].split(".")[0]+".png")
            visualized_output.save(out_path)
            np.save(os.path.join(args.output, path.split("/")[-1].split(".")[0]+".npy"), predictions)
            
            kernel = np.ones((5, 5), np.uint8)

            if predictions["instances"].scores.sum()!=0:
                mask = copy.copy(img)
                mask[:] = 0
                img = copy.copy(img)
                if len(predictions["instances"])==1:

                    try:
                        original = predictions['instances'].pred_masks.int().detach().cpu().numpy().reshape(predictions['instances'].pred_masks.shape[1],  predictions['instances'].pred_masks.shape[2]).astype(np.uint8)
                    except:
                        original = prediction['instances'].pred_masks.int().detach().cpu().numpy().reshape(prediction['instances'].pred_masks.shape[0],  prediction['instances'].pred_masks.shape[1]).astype(np.uint8)
                    erosion = cv2.erode(original, kernel=kernel, iterations=5)
                    dilation = cv2.dilate(original, kernel=kernel, iterations=3)
                    diff = dilation-erosion
                    diff = diff.astype(bool)
                    mask[diff] += 50
                else:
                    for i in range(len(predictions["instances"])):
                        original = predictions['instances'].pred_masks[i].int().detach().cpu().numpy().astype(np.uint8)
                        
                        erosion = cv2.erode(original, kernel=kernel, iterations=5)
                        dilation = cv2.dilate(original, kernel=kernel, iterations=3)
                        diff = dilation-erosion
                        diff = diff.astype(bool)
                        mask[diff] += 50
                blur_kernel = np.ones((5,5),np.float32)/25
                mask = cv2.filter2D(mask, -1, blur_kernel)
                cv2.imwrite(os.path.join(args.output, path.split("/")[-1].split(".")[0]+".grasp.png"), img+mask)
            else:
                print(predictions["instances"].pred_masks)
           
    
    
    '''
python demo_imgs.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input /home/nagata/sources/Detic/demo_inputs/ --output /home/nagata/sources/Detic/demo_outputs/ --vocabulary lvis --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

    '''