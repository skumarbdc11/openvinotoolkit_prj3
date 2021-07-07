'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from utils import ModelBase

class Model_GazeEstimation(ModelBase):
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.6):
        super(Model_GazeEstimation, self).__init__(model_name, device, extensions, threshold)

    def predict(self, leyeframe, reyeframe, hpangles):
        self.input_shape = self.network.inputs["left_eye_image"].shape
        outputs = self.run_infer(leyeframe, reyeframe, hpangles)
        return self.preprocess_output(outputs[self.output_blob])

    def preprocess_output(self, outputs): 
        # Shape: 1x3
        return(outputs[0])

    def set_framedims(self, width, height):
        ## Input Frame dimensions
        ## Only for visualization demo purpose
        ## center point of the media frame has been chosen as beginning point
        self.mx = int(width / 2)
        self.my = int(height / 2)
        
    def get_movement(self, gx, gy, w, h):
        ## Only for visualization demo purpose for approximate mouse movement
        ## Update the existing co-ordinates for showing the movement
        self.mx += int(gx * w / 300)
        self.my -= int(gy * h / 300)
        return self.mx, self.my
