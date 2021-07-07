'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np
from utils import ModelBase

class Model_FaceDetection(ModelBase):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.6):
        super(Model_FaceDetection, self).__init__(model_name, device, extensions, threshold)

    def predict(self, image):
        outputs = self.run_infer(image)
        ## Process the output
        self.face_frame = None
        face_box = self.preprocess_output(outputs[self.output_blob], image)
        return face_box

    def preprocess_output(self, outputs, image):
        ## Process the output of the Model as per the expected output format
        ## Shape 1x1xNx7
        ## The net outputs blob with shape: [1, 1, N, 7], where N is the number of detected bounding boxes. 
        ## Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max]
        for box in outputs[0][0]:
            conf = box[2]
            if conf >= self.threshold:
                self.fid = box[0]
                ## Calculate the box coords against frame size
                x1, y1 = int(box[3] * self.fwidth), int(box[4] * self.fheight)
                x2, y2 = int(box[5] * self.fwidth), int(box[6] * self.fheight)
                ## Extract the face frame with facebox details retrieved from the inference engine
                self.face_frame = image[y1:y2, x1:x2]
                ## Considering only the first face identified, so just return only that.
                return np.array([x1,y1,x2,y2])

    def get_faceframe(self):
        ## Face frame captured post inference on media frame
        return self.face_frame
    
    def get_faceid(self):
        ## Only for debugging puprose
        return self.fid
    
    def set_framedims(self, width, height):
        ## Input Frame dimensions to normalize the facebox co-ordinates
        self.fwidth = width
        self.fheight = height
