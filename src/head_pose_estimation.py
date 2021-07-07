'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from utils import ModelBase

class Model_HeadPoseEstimation(ModelBase):
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.6):
        super(Model_HeadPoseEstimation, self).__init__(model_name, device, extensions, threshold)

    def predict(self, image):
        outputs = self.run_infer(image)
        ## Process the output
        return self.preprocess_output(outputs)

    def preprocess_output(self, outputs):
        ## Process the output of the Model as per the expected output format
        # Shape 1x1
        # Output layer names in Inference Engine format: 
        #   name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees)
        #   name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees)
        #   name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees)
        angle_y_fc = outputs["angle_y_fc"][0][0]
        angle_p_fc = outputs["angle_p_fc"][0][0]
        angle_r_fc = outputs["angle_r_fc"][0][0]
        return angle_y_fc, angle_p_fc, angle_r_fc
