'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np
from utils import ModelBase

class Model_FacialLandmarksDetection(ModelBase):
    '''
    Class for the Facial Landmarks Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.6):
        super(Model_FacialLandmarksDetection, self).__init__(model_name, device, extensions, threshold)

    def predict(self, image, fwidth, fheight):
        outputs = self.run_infer(image)
        ## Process the output
        return self.preprocess_output(outputs[self.output_blob], fwidth, fheight)

    def preprocess_output(self, outputs, fwidth, fheight):
        ## Process the output of the Model as per the expected output format
        # Shape 1x10x1x1
        # The net outputs a blob with the shape: [1, 10], containing a row-vector of 10 floating point values for 
        # five landmarks coordinates in the form (x0, y0, x1, y1, ..., x5, y5)
        ## Calculate the left and right eye co-ordinates (which is the first pair of values in the output box)
        ## against the face frame size
        ## other co-ordinates are ignored for this use case
        self.x1, self.y1 = int(outputs[0][0][0][0] * fwidth), int(outputs[0][1][0][0] * fheight)
        self.x2, self.y2 = int(outputs[0][2][0][0] * fwidth), int(outputs[0][3][0][0] * fheight)
        return np.array([self.x1, self.y1, self.x2, self.y2])

    ## Set the eye box dimension for eye frame capture
    def set_eyedims(self, width, height):
        ## Store the eye box dimenstions for extracting the eyeframes later
        self.ewidth = width
        self.eheight = height
    
    ## Capture the eyeframe from the given x,y
    def get_eyeframes(self, faceframe):
        #  (x1,y1)
        #  |---------------|
        #  |               |
        #  |               |
        #  |       o(x,y)  |
        #  |               |
        #  |               |
        #  |---------------| (x2,y2)
        #
        # Where x,y is the co-ordinates received as part of inference 
        # and (x1,y1) & (x2,y2) with respect to the defined boundary for eyebox will be calculated here
        ## Bounding box co-ordinate calculation with the given x,y
        # Left Eye
        self.lx1, self.ly1 = self.x1-int(self.ewidth/2), self.y1-int(self.eheight/2)
        self.lx2, self.ly2 = self.x1+int(self.ewidth/2), self.y1+int(self.eheight/2)
        ## Take out the eyebox from face frame blob
        leyeframe = faceframe[self.ly1:self.ly2, self.lx1:self.lx2]
        # Right Eye
        self.rx1, self.ry1 = self.x2-int(self.ewidth/2), self.y2-int(self.eheight/2)
        self.rx2, self.ry2 = self.x2+int(self.ewidth/2), self.y2+int(self.eheight/2)
        ## Take out the eyebox from face frame blob
        return leyeframe, faceframe[self.ry1:self.ry2, self.rx1:self.rx2]

    ## Get Eye boxes
    def get_eyeboxes(self):
        return np.array([self.lx1, self.ly1, self.lx2, self.ly2, self.rx1, self.ry1, self.rx2, self.ry2])
