# Anti Deep Fake
A Deep Learning classification to detect faked videos from video inputs and render them as an exported output with annotaions of people's faces in the video as `real` or `fake`, using FaceForensics++ paper @ ~60% accuracy on HD videos
&nbsp;
![Tensorflow](https://miro.medium.com/max/512/0*1HWz9KQ-duZZykT7.png)

![Image](https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_14%2Fproject_333691%2Fimages%2Fimg%2Fdetection_pipeline.png)

This is an implmentation of XceptionNet from the paper trained on our FaceForensics++ dataset. Besides the full image models, all models were trained on slightly enlarged face crops with a scale factor of 1.3.
The models were trained using the Face2Face face tracker, though the `detect_from_models.py` file uses the freely available dlib face detector.

Note that we provide the trained models from our paper which have not been fine-tuned for general compressed videos. You can find our used models under [this link](https://github.com/pourabkarchaudhuri/anti-deep-fake/releases/download/1.0/antideepfake_models.1.zip).   



### Install Python 3.6


##### Run package installer

```sh
$ pip install -r requirements.txt
```
Setup:
- Install required modules via `requirement.txt` file
- Run detection from a single video file or folder with
```shell
python detect_from_video.py
-i <path to input video or folder of videos with extenstion '.mp4' or '.avi'>
-m <path to model file, default is imagenet model
-o <path to output folder, will contain output video(s)
```  
from the classification folder. Enable cuda with ```--cuda```  or see parameters with ```python detect_from_video.py -h```.
