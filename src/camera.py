import cv2
from base_camera import BaseCamera
import model_load as od
import tensorflow as tf
import numpy as np
import sys

from object_detection.utils import visualization_utils as vis_util

class Camera(BaseCamera):
    video_source = 0

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        with od.detection_graph.as_default():
            with tf.Session(graph=od.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = od.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = od.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = od.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = od.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = od.detection_graph.get_tensor_by_name('num_detections:0')

                while True:
                    # Read current frame
                    _, image_np = camera.read()

                    #img_resized = cv2.resize(image_np, (480,360))

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        od.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)

                    # Encode as a jpeg image and return it
                    yield cv2.imencode('.jpg', image_np)[1].tobytes()
