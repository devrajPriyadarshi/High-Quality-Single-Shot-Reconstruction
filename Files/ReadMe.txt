View1.png and View2.png are the 2 images taken from Camera1 and Camera2 respectively
The Intrinsic and extrinsic parameters of the cameras are in CameraCalibration.txt

Further, the calibration parameters given can be combined to 
form projection matrices (3x4) for each camera such as:

Projection matrix for cam 1 (3x4) = K * [R1|t1]
And Projection matrix for cam 2 (3x4) = K * [R2|t2]