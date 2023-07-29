# Image segmentation algorithm ROS implementation
This is a ROS package utilizing a trained CNN for coastline detection through a ZED stereo camera inside the UAV simulator synthetic environment presented in https://github.com/sotomotocross/UAV_simulator_ArduCopter.git

The CNN implemented on the present ROS package is trained according to the pipeline https://github.com/sotomotocross/coast_cnn_based_detection_pipeline.git

## The ROS workspace setup
You have to create a separate catkin_ws to run the present package with the trained CNN. This ROS worskpace needs to be build and run using Python3. For this reason the user has to setup a python3 virtual environment and installing a set of dependencies inside. Also if you are an NVIDIA user then you have to check the CUDA versions, and cuDNN. Else you will be running the NN prediction on your CPU (either way the payload to the computer is heavy but GPU acceleration helps a lot).
The basic dependencies for the python3 are installed using the bellow commands:
```
$ cd ~
$ sudo apt update
$ sudo apt upgrade
$ sudo apt install build-essential cmake unzip pkg-config
$ sudo apt install libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
$ sudo apt install libjpeg-dev libpng-dev libtiff-dev
$ sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
$ sudo apt install libxvidcore-dev libx264-dev
$ sudo apt install libgtk-3-dev
$ sudo apt install libopenblas-dev libatlas-base-dev liblapack-dev gfortran
$ sudo apt install libhdf5-serial-dev
$ sudo apt install python3-dev python3-tk python-imaging-tk
```
After installing the dependencies you have to install Anaconda 3 (https://docs.anaconda.com/anaconda/install/linux/) for the Ubuntu 18.04 version.
After the Anaconda installation you create an anaconda virtual environment with the following terminal commands:
```
$ source ~/anaconda3/etc/profile.d/conda.sh
$ conda create -n tf-gpu-cuda10 tensorflow-gpu=1.14 cudatoolkit=10.0 python=3.6
$ conda activate tf-gpu-cuda10
$ conda install -c conda-forge keras=2.2.5
$ pip install keras-segmentation
```
Now you have to be inside the virtual environment so you have to install all the pip dependencies:
```
$ pip install numpy
$ pip install scipy matplotlib pillow
$ pip install imutils h5py==2.10.0 requests progressbar2
$ pip install cython
$ pip install scikit-learn scikit-build scikit-image
$ pip install opencv-contrib-python==4.4.0.46
$ pip install tensorflow-gpu==1.14.0
$ pip install keras==2.2.5
$ pip install opencv-python==4.4.0.42
$ pip install keras-segmentation
$ pip install rospkg empy
```
Now you have to check the import of the keras and tensorflow:
```
$ python
$ >>> import tensorflow
$ >>>
$ >>> import keras
$ Using TensorFlow backend.
$ >>>
$ >>> import keras_segmentation
$ >>>
```
If everything succesful you can check your python 3 virtual environment running the image-segmentation-keras tutorial (https://github.com/divamgupta/image-segmentation-keras) with the dataset given from the framework. It takes about 4-6 hours (depending on the PC's we have tested till this day) so you can leave at night. If all the predictions are ok then your python 3 virtual environment is ready for use.

The setup of the ROS workspace will be given below.
```
$ mkdir -p ~/image_seg_catkin_ws/src
$ cd ~/image_seg_catkin_ws
$ pip3 install numpy
$ pip3 install scipy matplotlib pillow
$ pip3 install imutils h5py==2.10.0 requests progressbar2
$ pip3 install cython
$ pip3 install scikit-learn scikit-build scikit-image
$ pip3 install opencv-contrib-python==4.4.0.46
$ pip3 install opencv-python==4.4.0.42
$ pip3 install rospkg empy
$ cd src
$ git clone https://github.com/OTL/cv_camera.git
$ git clone -b melodic-devel https://github.com/ros/geometry2.git
$ git clone https://github.com/ros-perception/image_common.git
$ git clone https://github.com/amc-nu/RosImageFolderPublisher.git
$ git clone -b melodic https://github.com/ros-perception/vision_opencv.git
$ rosdep install --from-paths src --ignore-src -r -y
$ catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
```
Now you can move all the ecatkin_ws content from the repo to the src directory of the ecatkin_ws you just created and build.
Then you execute the commands below:
```
$ cd ~/image_seg_catkin_ws
$ rosdep install --from-paths src --ignore-src -r -y
$ catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
$ source devel/setup.bash
```

While the simulator is running you open a terminal and run the commands below that can start the trained NN for coastline detection running:
```
$ cd ~/image_seg_catkin_ws
$  source devel/setup.bash
$ source ~/anaconda3/etc/profile.d/conda.shconda activate tf-gpu-cuda10
$ rosrun img_seg_cnn vgg_unet_predict.py
```
