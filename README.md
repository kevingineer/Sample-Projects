# Simple CNN classifier program - toys classifier

## Overview

Convolutional neural networks are deep artificial neural networks that are used primarily to classify images (e.g. name what they see), cluster them by similarity (photo search), and perform object recognition within scenes. They are algorithms that can identify faces, individuals, street signs, tumors, platypuses and many other aspects of visual data (Skymind, n.d).

For this sample project, a program will be trained to classify objects that will fall among the following classes:
1. Dino (Dilophosaurus from the film franchise Jurassic Park)
2. Lugia (Legendary pocket monster from the anime franchise Pokémon)
3. Taz (Tasmanian Devil from the cartoon Looney Tunes)

A recognition (classifier) program called “server” will create a model that is trained to classify the toy objects. Meanwhile, a program called "client" will process new images and classify them using the model created by "server".


## Dependencies
Tthe scripts that were used to train and test the Toys classifier ran successfully on the environment as described below:

**System:**
- OS : Windows 10 
- CPU: Intel (R) Core(TM) i3-4170 CPU @ 3.70GHz 
- System Type: 64-bit OS, x64-based processor
- GPU : NVIDIA GeForce GTX 760 Ti 1 GB
- RAM: 8GB

**Software versions:**
- Python 3.5 
- Python Libraries(latest compatible versions, all installed using native pip):
  OpenCV (cv2),
  tqdm (progress bar),
  Tensorflow (GPU),
  TFlearn,
  matplotlib
- CUDA® Toolkit 9.0
- cuDNN v7.0


## Execution Guide
**A. Prepare the environment. Below is a guide on how to re-create the enviroment used to create and train the Toys model.**
  1. Download and install Python 3.5. Option to install pip under "Optional Features" should be checked. (https://www.python.org/downloads/release/python-350/)
  
  2. Download and install CUDA Toolkit 9.0 (https://developer.nvidia.com/cuda-90-download-archive)
  
  3. Download and install cuDNN v7.0: 
     - Click on this link (https://developer.nvidia.com/rdp/form/cudnn-download-survey) and login the user's Nvidia credentials or create an new account.
     - After logging in, user should be in the cuDNN download page https://developer.nvidia.com/rdp/cudnn-download
     - Accept the Terms and Conditions and click "Archived cuDNN Releases"
     - User should be in the cuDNN archive page (https://developer.nvidia.com/rdp/cudnn-archive)
     - Find and click the "Download cuDNN v7.0.5 (Dec 5, 2017), for CUDA 9.0". A list will pop-up. 
     - Look for cuDNN v7.0.5 Library for Windows 10 and click it. 
     - A file named cudnn-9.0-windows10-x64-v7.zip should start being downloaded afterwards.
     - Unzip the file and copy the following files into the CUDA Toolkit directory: 
Copy <installpath>\cuda\bin\cudnn64_7.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin.
Copy <installpath>\cuda\ include\cudnn.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include.
Copy <installpath>\cuda\lib\x64\cudnn.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64.
     
  4. Install the required python libraries:
     - Open command prompt(cmd) or powershell as Administrator.
     - Run the codes below:
     - pip3 install --upgrade tensorflow-gpu
     - pip3 install opencv-python
     - pip3 install tqdm
     - pip3 install tflearn
     - pip3 install matplotlib
   
  5. Create a local folder where the user will place the required files for this sample project.
   
  6. For an alternative guide to install Tensorflow, user may refer to the TF site https://www.tensorflow.org/install/
    
**B. After preparing the environment:**
  1. Download the Training dataset: https://drive.google.com/file/d/1Af0lRPzE9Ej6AFhIOXXOUDj9IefO9rOf/view?usp=sharing
  2. Download the Test dataset: https://drive.google.com/file/d/1KPREOoK2VR45iUh1jbuHdt4_ymeaBkR4/view?usp=sharing
  3. Unzip the files and save them on the local folder created on step A.5 with folder name 'Train' and 'Test' respectively.
  
**C. Prepare the classifier program:**
  1. Download the Python scripts "server" and "client" saved on this repository: https://github.com/kevingineer/Sample-Projects
  2. Place the scripts on the local folder created on step A.5.
  3. Edit the two scripts and change the value of the TRAIN_DIR and TEST_DIR to reflect the 'Train' and 'Test' folders on step B.3.
  4. Save the changes on the scripts. 
 
**D. Run the "server" script.**
  1. Using command prompt (cmd) or Powershell, sample code:
     C:\user>python "C:\User\Toys\client.py"
  2. If using IDLE Python, open the script "client.py" and press F5.
  3. The model should start training and output files will be created: model, toys_train_data.npy, logs, etc.
 
**E. Run the "server" script.**
  1. Using command prompt (cmd) or Powershell, sample code:
     C:\user>python "C:\User\Toys\server.py"
  2. If using IDLE Python, open the script "server.py" and press F5.
  3. The model should start classifying the images and output Figure plots will be created : Figure 1 and Figure 2. Output files will be created: output-file.csv, toy_test_data.npy, etc.
  
** **Note that when using CPU-only Tensorflow, the following code must be run on command prompt (cmd) or shell pip3 \
     :install --upgrade tensorflow. Step 2 and 3 should be skipped. 
  
  
## References:
- Convolutional Neural Network (CNN), A.I. Wiki.Retrieved from http://www.psu.edu/ur/about/nittanymascot.htmlhttps://skymind.ai/wiki/convolutional-network
- Python : https://www.python.org/
- Tensorflow: https://www.tensorflow.org/
- cuDNN :https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
- Nvidia Developer: https://developer.nvidia.com/
