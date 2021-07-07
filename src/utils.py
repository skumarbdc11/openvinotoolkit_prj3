import cv2
import os.path
import logging as log
import numpy as np
import time
from openvino.inference_engine import IECore, IENetwork

from sys import platform
## Video Codec configuration
CODEC = cv2.VideoWriter_fourcc('F','M','P','4')

## Video Writer Wrapper class
class VideoWriter:
    def __init__(self, verbose, filename, fps, w, h):
        self.verbose = verbose
        if verbose:
            self.writer = cv2.VideoWriter(filename, CODEC, fps, (w, h), True)

    def rectangle(self, frame, *args):
        if self.verbose:
            cv2.rectangle(frame, (args[0], args[1]), (args[2], args[3]), args[4], args[5])

    def circle(self, frame, *args):
        if self.verbose:
            cv2.circle(frame, args[0], args[1], args[2], thickness=2)

    def text(self, frame, *args):
        if self.verbose:
            cv2.putText(frame, args[0], args[1], cv2.FONT_HERSHEY_COMPLEX, args[2], args[3], args[4])

    def write(self, frame):
        if self.verbose:
            self.writer.write(frame)

    def release(self):
        if self.verbose:
            self.writer.release()
            
    def waitKey(self, key):
        cv2.waitKey(key)

## Model base class
class ModelBase:
    def __init__(self, model_name, device, extensions, threshold):
        log.info("Initializing the model {}".format(model_name))
        ## Initializing all instance variables
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.threshold = threshold

        ## Others to Defaults
        self.core = self.network = self.net = self.input_blob = self.input_shape = self.output_blob = self.output_shape = None

        ## Check for unsupported Layer
        model_weights = os.path.splitext(self.model_name)[0] + ".bin"
        model_structure = self.model_name
        self.core = IECore()
        log.debug("Reading network...")
        self.network = self.core.read_network(model=model_structure, weights=model_weights)

        # Add a CPU extension, if applicable
        if self.extensions and "CPU" in self.device:
            log.debug("Adding extensions...")
            self.core.add_extension(self.extensions, self.device)

        ## Check for any unsupported layers
        supported_layers = self.core.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers)!=0 and self.device=='CPU':
            log.error("Model has unsupported layers, exiting")
            exit(1)
        log.debug("No unsupported layers")

        self.input_blob = next(iter(self.network.inputs))
        log.debug("[ Input Properties {} ]".format(self.input_blob))
        self.input_shape = self.network.inputs[self.input_blob].shape
        log.debug("[ Input Shapes {} ]".format(self.input_shape))
        self.output_blob = next(iter(self.network.outputs))
        log.debug("[ Output Properties {} ]".format(self.output_blob))
        self.output_shape = self.network.outputs[self.output_blob].shape
        log.debug("[ Output Shapes {} ]".format(self.output_shape))

    def load_model(self):
        log.info("Loading the model {}".format(os.path.basename(self.model_name)))

        ## Loading the Model
        start_time = time.time()
        self.net = self.core.load_network(network=self.network, device_name=self.device,num_requests=1)
        
        ## Model loading time
        log.debug("Model {} Loading time {:.2f} msecs".format(os.path.basename(self.model_name), (time.time()-start_time)*1000))

    def preprocess_input(self, image):
        frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        frame = frame.transpose((2,0,1))
        frame = frame.reshape(1, *frame.shape)
        return frame

    def run_infer(self, *params):
        ## Preprocess the input
        if len(params) == 1:
            input_frame = self.preprocess_input(params[0])
            ## Run the inference
            start_time = time.time()
            outputs = self.net.infer({self.input_blob:input_frame})
        elif len(params) == 3:
            ## Gaze estimation model has different set of inputs
            left_eye_frame = self.preprocess_input(params[0])
            right_eye_frame = self.preprocess_input(params[1])
            ## Run the inference
            start_time = time.time()
            outputs = self.net.infer({"left_eye_image": left_eye_frame,"right_eye_image": right_eye_frame,"head_pose_angles": [[*params[2]]]})
        else:
            log.error("Invalid parameters for running inference")
            exit(1)

        log.debug("Model {} Inference time {:.2f} msecs".format(os.path.basename(self.model_name), (time.time()-start_time)*1000))
        
        return outputs
