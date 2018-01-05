# body_poses
This program takes a video input from webcam or video file predict poses using torso, face and hand detection.
This program helps to identify 9 different human body poses  with a frame rate of upto 15 frames per second on a i-5 4110 CPU.
It succesfully predict 9 different poses with an acuracy of more then 80%.
Required softwares for the program to run:
1. Opencv (2.4.13)
2. Numpy (latest)
3. Sklearn (optional if you want to use the training model that has been uploaded)
4. Pickle

Place all the files in a folder.
After installing these libraries/sofwares run detect_poses.py
Make appropiate changes in VideoCapture function (which type of input you want '0' for webcam and 'filename' for  recorded video feed).
