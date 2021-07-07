"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Application main entry point for Computer Pointer Controller

Pre-Trained models used:
Face Detection: https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html
Facial Landmarks: https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html
Head Pose estimation: https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html
Gaze detection: https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


import os
import logging as log
import numpy as np
import cv2

## Models used in this application
from face_detection import Model_FaceDetection
from facial_landmarks_detection import Model_FacialLandmarksDetection
from head_pose_estimation import Model_HeadPoseEstimation
from gaze_estimation import Model_GazeEstimation

## Utilize the MouseController as output
from mouse_controller import MouseController

## Utilize the inputfeeder for reading recorded video or camera input
from input_feeder import InputFeeder

from utils import VideoWriter

from argparse import ArgumentParser
from sys import platform

## Constants
CAMERA = "cam"
VIDEO = "video"
OUT_VIDEO = "preview_output.mp4"

## Argument Parser
def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--media", required=True, type=str, 
                        help="Path to the recorded video file or cam for live web camera input")
    parser.add_argument("-fd", "--face_detection_model", required=True, type=str,
                        help="Model file for Face Detection with absolute path")
    parser.add_argument("-fl", "--facial_landmarks_model", required=True, type=str,
                        help="Model file for Facical Landmarks Detection with absolute path")
    parser.add_argument("-hp", "--head_pose_model", required=True, type=str,
                        help="Model file for Head Pose Estimation with absolute path")
    parser.add_argument("-ge", "--gaze_estimation_model", required=True, type=str,
                        help="Model file for Gaze Estimation with absolute path")
    parser.add_argument("-d", "--device", required=False, type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-e", "--cpu_extension", required=False, type=str,default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the "
                             "kernels impl.")
    parser.add_argument("-pt", "--prob_threshold", required=False, type=float, default=0.6,
                        help="Probability threshold for detections filtering (0.6 by default)")
    parser.add_argument("-l", "--log_level", required=False, type=str, default="INFO",
                        help="Application log level INFO | DEBUG | ERROR")
    parser.add_argument("-o", "--output_video", required=False, type=str, default=OUT_VIDEO,
                        help="Preview output video file")
    parser.add_argument("-v", "--verbose_preview", required=False, action='store_true',
                        help="Flag to show the visual indications on the output video")
    parser.add_argument("-s", "--live_show", required=False, action='store_true',
                        help="Flag to show the live frame")

    return parser

## Main Pipeline
def PipeLine(args, *models):
    ## Input Media Handling
    mediaFile = args.media
    inputFeeder = None
    if mediaFile.lower()==CAMERA:
            inputFeeder = InputFeeder(CAMERA)
    else:
        if not os.path.isfile(mediaFile):
            log.error("Unable to locate the media file")
            exit(1)
        inputFeeder = InputFeeder(VIDEO,mediaFile)

    ## Load the input media feeder
    inputFeeder.load_data()

    ## Initialize the video writer
    fps, w, h = inputFeeder.get_props()
    log.debug("FPS: {}, Width:{}, Height:{}".format(fps, w, h))
    out = VideoWriter(args.verbose_preview, OUT_VIDEO, fps, w, h)
    
    ## Construct the Mouse Controller
    mousecontrol = MouseController('medium', 'medium')

    ## Start the media frame loop
    log.info("Starting the Media frame Loop...")
    
    ## Face Detection Model
    fd = models[0]
    fd.set_framedims(w, h) ## required for extracting the output box co-ordinates
    
    ## Head Pose estimation Model
    hp = models[1]
    
    ## Face Landmarks Detection Model
    fld = models[2]
    
    ## Gaze estimation Model
    ge = models[3]
    if args.verbose_preview:
        ge.set_framedims(w,h) ## for visual perspective

    if args.live_show:
        cv2.namedWindow('live_show', cv2.WINDOW_AUTOSIZE)

    fc = 0
    for ret, frame in inputFeeder.next_batch():
        if not ret:
            break

        key = out.waitKey(60)

        fc += 1

        ## Face Detection Model Inference: Input{Media Frame}, Output{Face Box}
        ### <= facebox holds the box area for the first face detected
        ### <= the parameters are bound to the original media frame size
        facebox = fd.predict(frame)  # get the face box [x1,y1,x2,y2]
        # get the face frame which will be used as input for HP and FLD models
        ### <= faceframe is the image blob for the facebox
        faceframe = fd.get_faceframe()
        if faceframe is None: # If no face detected, continue in the media frame loop
            if key==27:
                break
            out.write(frame)
            continue
        ## Highlight the face identified - only if -v option is selected
        if args.verbose_preview:
            log.debug("ID:{} ({},{},{},{})".format(fd.get_faceid(),facebox[0], facebox[1], facebox[2], facebox[3]))
            out.rectangle(frame, facebox[0], facebox[1], facebox[2], facebox[3], (0,234,234), 2)

        ## Head Pose Estimation Model Inference: Input{Face frame}, Output{Head Pose Angles}
        ### <= hpangles are the YPR angles of the head position within faceframe
        hpangles = hp.predict(faceframe)
        ## Write the text on the frame for hp angles - only if -v option is selected
        if args.verbose_preview:
            hptext = "Head Pose Angles: Y:{:.2f} | P:{:.2f} | R:{:.2f}".format(hpangles[0], hpangles[1], hpangles[2]);
            #log.debug(hptext)
            out.text(frame, hptext, (20, 20), 1, (255, 0, 0), 2)
        
        ## Facial Landmarks Detection Model Inference: Input{Face frame}, Output{Left (x0,y0) and Right Eye (x1,y1)}
        # Set the face frame size to the Model to normalize the eye co-ordinates
        ## <= facewidth, faceheight represents the size of the facebox
        facewidth, faceheight = (facebox[2] - facebox[0]), (facebox[3] - facebox[1])
        # Now run the inference on the Face frame
        ## <== eyepoints holds just the x,y co-ordinates for both left and right eye with the facebox
        eyepoints = fld.predict(faceframe, facewidth, faceheight)

        ## Gaze Estimation Model inference: Input{eyeframes, headposition angles}, Output{x,y, gaze_vector}
        ## First lets take the eyeframe images:
        ## Make eyebox size with some specific value
        fld.set_eyedims(40,30) ## <= 40x30 is considered as eyebox boundary for this application
        ## Now get the eyeboxes with respect to the dimensions set above
        ## Note: that now the frame dimension is still set with facebox and 
        ## the eyebox should be aligned to that for gaze estimation model input
        leyeframe, reyeframe = fld.get_eyeframes(faceframe)
        if not leyeframe.size or not reyeframe.size:
            continue
        ## Draw circle on the eye - only if -v option is selected
        ## Add the facebox co-ordinates so that it will be normalized for the full media frame
        if args.verbose_preview:
            ## Calculate the eyepoint with respect to the media frame
            ## for this, use the facebox co-ordinates
            ebox = fld.get_eyeboxes()
            out.rectangle(frame, (facebox[0] + ebox[0]), (facebox[1] + ebox[1]), (facebox[0] + ebox[2]), (facebox[1] + ebox[3]), (255,0,0), 2)
            out.rectangle(frame, (facebox[0] + ebox[4]), (facebox[1] + ebox[5]), (facebox[0] + ebox[6]), (facebox[1] + ebox[7]), (255,0,0), 2)
        
        ## Now run the Gaze Estimation Model inference
        gaze_x, gaze_y, gaze_vector = ge.predict(leyeframe, reyeframe, hpangles)
        ## Draw gaze pointer on the frame - only if -v option is selected
        if args.verbose_preview:
            gx, gy = ge.get_movement(gaze_x, gaze_y, w, h)
            #log.debug("GX:{}, GY:{}".format(gx,gy))
            out.circle(frame, (gx,gy), 8, (255,255,255))
        
        if args.live_show and fc%2==0:
            cv2.imshow("live_show",cv2.resize(frame, (600,480)))

        ## Sending the output of Gaze Estimation co-ordinates to Mouse controller
        if fc%2==0:
            mousecontrol.move(gaze_x,gaze_y)

        ## Verbose preview on output video
        if args.verbose_preview:
            out.write(frame)
        
        if key==27:
            break
        ### Media Frame Loop

    ## Application cleanup
    log.info("End of Media frame Loop.")
    if args.output_video:
        out.release()
    inputFeeder.close()

## Application Main entry
def main(args):
    ## Construct the given models
    face = Model_FaceDetection(args.face_detection_model, args.device, args.cpu_extension, args.prob_threshold)
    face.load_model()

    facelandmarks = Model_FacialLandmarksDetection(args.facial_landmarks_model, args.device, args.cpu_extension, args.prob_threshold)
    facelandmarks.load_model()

    headpose = Model_HeadPoseEstimation(args.head_pose_model, args.device, args.cpu_extension, args.prob_threshold)
    headpose.load_model()

    gazeestimation = Model_GazeEstimation(args.gaze_estimation_model, args.device, args.cpu_extension, args.prob_threshold)
    gazeestimation.load_model()

    ## Everthing is fine now, we can start the pipeline
    PipeLine(args, face, headpose, facelandmarks, gazeestimation)

if __name__ == '__main__':
    ## Grab command line args
    args = build_argparser().parse_args()

    ## Initialize logger
    log.basicConfig(level=args.log_level, format="%(asctime)s:%(levelname)s: %(message)s")

    ## Check the Model existence
    for modelFile in [args.face_detection_model, args.facial_landmarks_model, args.gaze_estimation_model, args.head_pose_model]:
        if not os.path.isfile(modelFile):
            log.error("Unable to locate specified model file {} ".format(modelFile))
            exit(1)

    main(args) 
