#!/usr/bin/env python
import rospy
import tensorflow as tf
from keras_segmentation.predict import model_from_checkpoint_path
from detection_processor import DetectionProcessor


def main():
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_virtual_device_configuration( gpus[0],
                                                                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2500)])
            # tf.config.experimental.set_virtual_device_configuration( gpus[0])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU \n")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    gpu = tf.test.gpu_device_name()

    # Check available GPU devices.
    print("The following GPU devices are available: %s" % tf.test.gpu_device_name())
    
    global mdl
    mdl = model_from_checkpoint_path("src/img_seg_cnn/model_checkpoints/mobilenet_segnet/mobilenet_segnet224")
    # Retrieve the model checkpoint path from the ROS parameter server
    
    # model_checkpoint_path = rospy.get_param('~model_checkpoint_path', '')

    # global mdl
    # mdl = model_from_checkpoint_path(model_checkpoint_path)
    
    global graph
    graph = tf.get_default_graph()
    # Initialize the ROS node with a unique name
    rospy.init_node('detection_node', anonymous=True)

    # Create an instance of DetectionProcessor
    detection_processor = DetectionProcessor(mdl, graph)

    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    main()