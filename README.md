# Implicit-IoT-Authentication-Using-On-Phone-ANN-Models-and-Breathing-Data

Here we build an example demo that illustrates a possible deployment of a breathing based authentication model. The android app is based off of speech_commands example in [TensorFlow’s examples](https://github.com/tensorflow/examples/tree/master/lite/examples) and we thank them for their in depth templates.

These were run locally and we try to collect things together in the repo for historical purposes. The structure is set up as the following:

1. Model_Trainning - where we used processed audio sound files to train models
2. Server - where we use a local computer to facilitate authentication results from the phone to the Fitbit device
3. TFL_Auth - the android application using trained models to access user authentication validity based off breathing audio


-Will Cheung