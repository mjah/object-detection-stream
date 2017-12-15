# Object Detection Stream

This project allows you to stream a live video feed of objects being detected.

TensorFlow is used with the help of a pre-trained model to detect objects in a live video feed. The TensorFlow Object Detection API is used alongside the SSD Mobilenet v1 Coco model, this pretrained model is one of the fastest to detect objects (as of late 2017).

## Main Dependencies

* [Tensorflow](https://www.tensorflow.org/)
* [Flask](http://http://flask.pocoo.org/)
* [OpenCV](https://opencv.org/)
* [Python](https://www.python.org/)

## Running the Application

Install _pipenv_ if not installed:

```sh
pip install -U pipenv
```

Set up the environment:

```sh
cd src/
pipenv install
pipenv shell
```

Then to run:

```sh
python app.py
```

## Preview

![Preview](/preview.png?raw=true)

## Helpful Resources

* [Flask Video Steaming Revisited - miguelgrinberg.com](https://blog.miguelgrinberg.com/post/flask-video-streaming-revisited)

* [Object Detection Demo - tensorflow/models](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb)
