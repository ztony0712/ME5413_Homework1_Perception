# AMR Perception
First of all, create a virtual environment and install the required packages using the following commands:
```pip install -r requirements.txt```

## Task1 and Task2

Put the jupyter notebooks back to the Homework1_Perception directory you provided, they will run without issues.
Otherwise, you will need to change the path to the data files in the notebooks. 

### Outputs

Task1 notebook produces 5 results videos and 5 average IoU plots.

Task2 notebook produces 'viz.png' for trajectories visualization, and 'metrics.png' for displacement errors on all time steps.

### Bonus Task

- Create and build a new ROS package. Put the ```detector.py``` and the ```viz.py``` in the 'scripts' directory. Change the content in the 'CMakeLists.txt' and 'package.xml' files accordingly.
- catkin_make the package and source the setup.bash file. Start the roscore with ```roscore```
- Change the file path of ```src/beginner_tutorials/data/<seq_num>/``` to the directory you put the groundtruth.txt and firsttrack.txt. 
- Run the ros node using the following command:
```rosrun <work_space_name> detector.py```
- If you want a visualization, run the corresponding node using the following command:
```rosrun <work_space_name> viz.py```
- Play the bag file using the following command:
```rosbag play <bagname>.bag```
