import numpy as np
from astronomaly.base.base_pipeline import PipelineStage
import tensorflow as tf
import numpy as np
import sys
sys.path.append('/Users/mrgr/Documents/GitHub/FiLeK/')
from filek.models import lens15
from filek.preprocessing import scaling_clipping

# Load the saved TensorFlow model from disk
model = lens15()
layer_name = 'Conv8'
# Get a handle to the desired layer's output tensor
layer_output = model.get_layer(layer_name).output


class TE_Features(PipelineStage):
    def __init__(self, 
                 model_choice=lens15,
                 **kwargs):
        """
        Runs a pretrained CNN and extracts the deep features before the 
        classification layer.

        Parameters
        ----------
        model_choice: string
            The model to use. Options are:
            'zoobot', 'resnet18' or 'resnet50'. These also use predefined
            transforms
        """

        super().__init__(model_choice=model_choice, **kwargs)

        self.model_choice = model_choice
        # Easiest to set these once this has been run once
        self.labels = []


    # Define a function that takes an image as input and returns the output of the desired layer
    def _execute_function_1(image):
        
        image = scaling_clipping(image)
        dim_idx = np.where(np.array(image.shape) == 4)[0]
        if dim_idx==0:
            image = np.transpose(image, (1,2,0))
        elif dim_idx==2:
            pass

        # Preprocess the image
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)

        # Preprocess the input image (assuming 'preprocess_input' is a function that preprocesses the input image for the model)
        #processed_image = preprocess_input(image)
        
        # Create a Keras model that outputs the desired layer's output tensor
        intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=layer_output)
        
        # Run the model on the input image and return the output
        feats = intermediate_model.predict(image)

        if len(labels) == 0:
            labels = [f'feat{i}' for i in range(len(feats))]
            
        return feats
