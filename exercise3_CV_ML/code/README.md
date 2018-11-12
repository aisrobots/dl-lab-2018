# Assignment 3 - semantic segmentation with fully convolutional networks

* For training first download and extract the training and testing data (described at `data/readme.md`)

* Training must be called with `Train_Net.py`

  `$ python Train_Net.py --checkpoint_dir=./checkpoints/ultraslimS_3/ --configuration=3`
  Here, `checkpoint_dir` must have the path for the checkpoints folder which stores parameters of the network as training    progresses.
  `configuration` is a flag for the type of decoder to be used.
  Please create a separate checkpoint directory for each individual experiment you run.

* After training the test part will use `Test_Net.py` which will generate a text file with all mean IoUs will be saved at: testIoU.txt

  `$ python  Test_Net.py --model_path=./checkpoints/ultraslimS=3 --configuration=3`
