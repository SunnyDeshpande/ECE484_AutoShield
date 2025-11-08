# ECE484_AutoShield
GEM Project by Team AutoShield for UIUC Fall '25 ECE484 Principles of Safe Autonomy



# Procedure 
#smart_extract.sh before build
password:
gem_ros2

# compile workspace
colcon build --symlink-install 

# launch sensor
source install/setup.bash
ros2 launch basic_launch sensor_init.launch.py

# launch corner cameras
source install/setup.bash
ros2 launch basic_launch corner_cameras.launch.py

# launch gnss
source install/setup.bash
ros2 launch basic_launch visualization.launch.py

# launch joystick control
source install/setup.bash
ros2 launch basic_launch dbw_joystick.launch.py

# launch path tracking controller, close joystick control first
source install/setup.bash
ros2 launch pacmod2 pacmod2.launch.xml
ros2 run gem_gnss_control pure_pursuit