# COMP3631 ROS Project

## How the project was run on lab machine

Three terminals are needed, one for the gazebo world, one for RViz navigation, one for project python script

---

### Terminal 1: Gazebo world

```bash
cd ~/ros2_ws
ros2 launch turtlebot3_gazebo turtlebot3_task_world_2026.launch.py
```
Wait until the Gazebo world had fully loaded
---

### Terminal 2: Nav2 with RViz

```bash
cd ~/ros2_ws
ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=True map:=$HOME/ros2_ws/src/ros2_project_sc23as2/map/map.yaml
```
RViz opened alongside Nav2. Before running the project we set the initial pose, using 2d pose estimate fromt the RViz toolbar on top, nav2 needs this to know where the robot is on the map, otherwise it will reject all navigation goals.

1. Click **2D Pose Estimate** in the RViz toolbar
2. Click on the map at the robot's spawn position
3. Hold and drag in direction the robot is facing and release

---

### Terminal 3: Project

Build the workspace from `~/ros2_ws`:
```bash
colcon build
# or just this package (faster)
colcon build --packages-select ros2_project_sc23as2
```

Then source the bash setup and run:
```bash
source install/setup.bash
ros2 run ros2_project_sc23as2 project
```
(On the lab machine I had this in ~/.bashrc but the setup is available in install/setup.bash)

---

## Project Runs

- Two OpenCV windows: the original live camera feed with coloured circles drawn around detected objects, and a filtered view showing only the detected colour pixels and black background
- The terminal prints 
  - waypoint progress, 
  - colour detection messages, 
  - and a final summary when the robot stops at the blue box
- The robot navigates through the waypoints using Nav2, and the moment it spots the blue box it drives directly toward it and stops approximately 1 metre away (by estimating big enough contour area of blue pixels)
