## Setup
1. Connect to the main server
      ```Shell
      ssh username@login.infor...freiburg.de
      ssh tfpool54
      ```
2. Login, and check if you can do
      ```Shell
      cd /project/ml_ws1819/username
      ```
If not, contact poolmgr@infor...
Subject: deep learning lab 2018 account, username and matrikelnr

3. Create a python3 virtual env and install libraries
      ```Shell
      virtualenv --system-site-packages -p python3 venv3
      source venv3/bin/activate
      pip install tensorflow-gpu # Optional: You can also use the pre-installed 1.3 version
      pip install tensorflow     # Optional: You can also use the pre-installed 1.3 version
      pip install hpbandster
      ```
## Running experiments

2. Do not run anything on the main server. Connect to one of the pool
  computer there instead. Preferably use tfpool25-46 or tfpool53-63 they provide the best GPUs.
      ```Shell
      ssh tfpoolXX
      ```
3. Replace XX with a 2 digit number of the computer you want to connect .
Before running make sure no one else is using the selected computer.
Also make sure your tensorflow program is running on the gpu and not on the cpu.
      ```Shell
      nvidia-smi
      who
      top
      ```      
4. Start a screen session
      ```Shell
      screen -S t01
      python3
      import tensorflow
      ```
5. Detach from screen using: ctrl+a+d  
6. Login back into screen
      ```Shell
      screen -ls
      screen -r t01
      ```
7. Write down on which computer you started your screen session.
