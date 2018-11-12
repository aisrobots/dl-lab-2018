#Assignment 3 - semantic segmentation with fully convolutional networks. 

For training first download the extract the training and testing file described at: 

data/readme.md


Training must be called with Train_Net.py

$ python Train_Net.py --checkpoint_dir=./checkpoints/ultraslimS=3/ --configuration=3

Where checkpoint_dir must have the path for the checkpoints folder and configuration
 is a flag for the type of decoder to be used.

After training the test part will use Test_Net.py which will generate a text file with 
all mean IoUs will be saved at: testIoU.txt

$ python  Test_Net.py --model_path=./checkpoints/ultraslimS=3 --configuration=3


