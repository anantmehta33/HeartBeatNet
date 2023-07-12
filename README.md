# HeartBeatNet
# Attention Based U-Net for PCG Signal classification <br>
![WhatsApp Image 2023-07-10 at 21 24 35](https://github.com/anantmehta33/HeartBeatNet/assets/87894541/ac31d1df-33a3-4c68-9119-d9383f93510d) <br>
# Methodology used in the pipeline: <br>
# Methodology: <br>

# U-Net Architecture: <br>
U-Net is a popular architecture for semantic segmentation tasks, which aims to classify each pixel in an image. It consists of an encoder (downsampling path) and a decoder (upsampling path). The encoder captures the context and extracts high-level features, while the decoder recovers spatial information and generates the final segmentation map.
In our code, the U-Net architecture is modified with additional fully connected layers at the end for classification purposes.

# Data Augmentation: <br>
Data augmentation is performed using the ImageDataGenerator class from TensorFlow. It applies various transformations such as shear, zoom, and horizontal flip to the training images. These transformations help increase the diversity and size of the training data, reducing overfitting and improving the model's generalization ability.

# Attention Mechanism: <br>
The attention mechanism in this code aims to enhance the model's ability to focus on informative regions in the input images.
The attention block takes two inputs: the feature maps from the U-Net decoder (x) and the feature maps from the corresponding encoder skip connection (gating).
The attention block performs the following steps:
            theta_x: A 2x2 convolution on the decoder feature maps (x) to reduce spatial dimensions.
            phi_g: A 1x1 convolution on the gating feature maps to reduce dimensions.
            upsample_g: Transposed convolution to upsample the gating feature maps to the same size as theta_x.
            concat_xg: Element-wise addition of upsampled gating and theta_x.
            act_xg: ReLU activation on the concatenated tensor.
            psi: 1x1 convolution to reduce the tensor depth to 1 (channel-wise attention).
            sigmoid_xg: Sigmoid activation to obtain attention coefficients in the range [0, 1].
            upsample_psi: Upsampling the attention coefficients to the original spatial dimensions.
            Element-wise multiplication between upsampled attention coefficients and the decoder feature maps (x).
            Result: 1x1 convolution to recover the original number of channels.
            Batch normalization is applied to the final result.
            
Model Training and Evaluation:
        The model is compiled with the categorical cross-entropy loss, SGD optimizer, and metrics including accuracy and sensitivity at a specific specificity.
        The fit method is used to train the model on the training set and validate it on the test set.
        The training history is stored in plotter.
        After training, the model makes predictions on the test set and calculates evaluation metrics such as the confusion matrix and classification report.
        Additionally, the code plots the Receiver Operating Characteristic (ROC) curve to assess the model's performance.

The attention mechanism in this code helps the model focus on relevant regions and can improve the segmentation results by assigning higher weights to informative features. It enhances the model's capability to capture fine details and boundaries, leading to better segmentation accuracy.

# The attention mechanism works as follows: <br>

# Inputs to the Attention Block: <br> 
        The attention block takes two inputs:
            x: Feature maps from the U-Net decoder, which capture high-level semantic information.
            gating: Feature maps from the corresponding encoder skip connection, which contain lower-level spatial information.

# Theta and Phi Convolutions: <br>
        Theta Convolution (theta_x): A 2x2 convolution is applied to the decoder feature maps (x) to reduce the spatial dimensions while preserving the channel                   information. This helps to capture the global context of the feature maps.
        Phi Convolution (phi_g): A 1x1 convolution is applied to the gating feature maps to reduce the dimensions while retaining the channel information. This                   operation reduces the computational complexity and prepares the gating feature maps for subsequent operations.

Upsampling the Gating Feature Maps:
        The gating feature maps are upsampled to the same spatial dimensions as the theta feature maps using a transposed convolution operation (upsample_g).
        This operation ensures that the gating feature maps have the same size as the theta feature maps, allowing element-wise addition later in the process.

Concatenation and Activation:
        The upsampled gating feature maps and the theta feature maps are concatenated (concat_xg).
        The concatenated tensor is passed through an activation function (ReLU) to introduce non-linearity and capture relevant relationships between the two sets of             feature maps (act_xg).

Attention Coefficients:
        The concatenated tensor is passed through a 1x1 convolutional layer (psi) to reduce the tensor depth to 1 (channel-wise attention).
        A sigmoid activation function is applied (sigmoid_xg) to obtain attention coefficients in the range [0, 1].
        These attention coefficients represent the importance of each channel in the gating feature maps for each spatial position in the theta feature maps.

Applying Attention to the Decoder Feature Maps:
        The attention coefficients are upsampled to the original spatial dimensions of the decoder feature maps (upsample_psi).
        The upsampled attention coefficients are repeated along the channel dimension to match the depth of the decoder feature maps using repeat_elem.
        Element-wise multiplication is performed between the upsampled attention coefficients and the decoder feature maps.
        This operation applies the attention weights obtained from the gating feature maps to the decoder feature maps, emphasizing important spatial locations and              features while suppressing less relevant information.

Result and Batch Normalization:
        The multiplied tensor is passed through a 1x1 convolutional layer (result) to recover the original number of channels.
        Batch normalization is applied to the resulting tensor (result_bn).
        Batch normalization helps stabilize the training process and improve the model's generalization by normalizing the tensor's statistics across the batch                   dimension.

By applying attention weights, the model can selectively enhance or suppress specific spatial locations and features, improving its ability to capture important details and boundaries during the segmentation 
process. <br>

# Workflow: <br>

![hello1](https://github.com/anantmehta33/HeartBeatNet/assets/71447155/99c9f47c-ab3e-48e1-8ce9-50969b61be27)




