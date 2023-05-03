import numpy as np
from astronomaly.base.base_pipeline import PipelineStage

try:
    import torch
    from torchvision import models
    from torchvision import transforms
except ImportError:
    err_string = "pytorch and torchvision must be installed to use this module"
    raise ImportError(err_string)

try:
    from zoobot.pytorch.training import finetune
    zoobot_available = True
except ImportError:
    zoobot_available = False


from astronomaly.preprocessing import image_preprocessing
import cv2
import tensorflow as tf
import sys
sys.path.append('/Users/mrgr/Documents/GitHub/FiLeK/')
#from filek import models
#from filek.preprocessing import scaling_clipping


print('zoobot available', zoobot_available)
class CNN_Features(PipelineStage):
    def __init__(self, 
                 model_choice='zoobot', 
                 zoobot_checkpoint_location='/Users/mrgr/Documents/GitHub/KiDS_astronomaly/example_data/zoobot/effnetb0_greyscale_224px.ckpt',
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
        
        # All the models use these
        default_transforms = [transforms.ToTensor(),
                              transforms.Resize(256, antialias=True),
                              transforms.CenterCrop(224)]
        # Normalizations used by resnet
        resnet_normalization = [transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])]

        if model_choice == 'zoobot':
            if not zoobot_available:
                err_string = "Please install zoobot to use this model: "
                err_string += "https://github.com/mwalmsley/zoobot"
                raise ImportError(err_string)

            if len(zoobot_checkpoint_location) == 0:
                err_string = ("Please download the weights for the zoobot" 
                              " model and provide the location of the" 
                              " checkpoint file in"
                              " zoobot_checkpoint_location")
                raise FileNotFoundError(err_string)

            self.transforms = transforms.Compose(default_transforms)

            self.model = finetune.load_pretrained_encoder(zoobot_checkpoint_location)
            #self.model = finetune.load_encoder(zoobot_checkpoint_location)
            print('using weights from zoobot')
        else:
            # It's one of the resnets
            transform_list = default_transforms + resnet_normalization

            self.transforms = transforms.Compose(transform_list)

            if model_choice == 'resnet18':
                wgts = models.ResNet18_Weights.DEFAULT
                model = models.resnet18(weights=wgts)
            else:
                wgts = models.ResNet50_Weights.DEFAULT
                model = models.resnet50(weights=wgts)

            # Strip off the last layer to get a normal feature extractor
            self.model = torch.nn.Sequential(*list(model.children())[:-1])

    def _execute_function(self, image):
        """
        Runs the appropriate CNN model

        Parameters
        ----------
        image : np.ndarray
            Input image

        Returns
        -------
        array
            Contains the extracted deep features
        """
        # The transforms can't handle floats to convert to uint8
        image = (image * 255).astype(np.uint8)
        #TODO WHY??
        '''
        if len(image.shape) == 2:  # Greyscale## 
            # Make a copy of this channel to all others
            image = np.stack((image,) * 3, axis=-1)
        '''
        
        #these models all want 3d images    
        # TE needs  input_shape = (101,101,4)
        # zoobot needs  input_shape = (101,101,1)
        dim_idx = np.where(np.array(image.shape) == 4)[0]
        if len(dim_idx)>0:
            if dim_idx==0:
                image = image[:3,:,:]
            elif dim_idx==2:
                image = image[:,:,:3]
        else:
            pass
        
        dim_idx = np.where(np.array(image.shape) == 3)[0]
        
        if dim_idx==2:
            pass
        elif dim_idx==0:
            image = np.transpose(image, (1,2,0))
        else:
            pass
        if self.model_choice== 'zoobot':
            #zoobot accepts only grey images
            #we just use the rband
            if len(image.shape)>2:
                #image = image[:,:,0]
                raise ValueError('Image has the wrong size for zoobot')
            
            #image = image_preprocessing.image_transform_greyscale(image)

        processed_image = self.transforms(image)
        
        # Add the extra alpha channel the nets expect
        processed_image = torch.unsqueeze(processed_image, 0)

        # Run the model, detach from the GPU, turn it into a numpy array
        # and remove superfluous dimensions (which will likely be a different)
        # number for different models
        feats = self.model(processed_image).detach().numpy().squeeze()

        if len(self.labels) == 0:
            self.labels = [f'feat{i}' for i in range(len(feats))]

        return feats
    

'''
class TE_Features(PipelineStage):
    def __init__(self, 
                 model_choice='lens15',
                 layer_name='Conv8',
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
        self.layer_name = layer_name
        # Easiest to set these once this has been run once
        self.labels = []


    # Define a function that takes an image as input and returns the output of the desired layer
    def _execute_function(self, image):

        if self.model_choice == 'lens15':
            model = models.lens15()

        # Get a handle to the desired layer's output tensor
        layer_output = model.get_layer(self.layer_name).output
        
        image = scaling_clipping(image)
        dim_idx = np.where(np.array(image.shape) == 4)[0]
        if len(dim_idx)==0:
            raise ValueError('These models need 4D images')
        if dim_idx==0:
            image = np.transpose(image, (1,2,0))
        elif dim_idx==2:
            pass

        # Preprocess the image
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)

        # Preprocess the input image (assuming 'preprocess_input' is a function that preprocesses the input image for the model)
        processed_image = scaling_clipping(image)
        
        # Create a Keras model that outputs the desired layer's output tensor
        intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=layer_output)
        
        # Run the model on the input image and return the output
        feats = intermediate_model.predict(image, verbose=0)

        if len(self.labels) == 0:
            labels = [f'feat{i}' for i in range(len(feats))]
            
        return feats
'''