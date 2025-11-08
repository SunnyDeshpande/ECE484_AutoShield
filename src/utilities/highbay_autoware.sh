echo "Starting can bus..."
sudo bash ~/Desktop/can_start.bash

echo "Starting Polaris Gem sensors..."
gnome-terminal -- sh -c "source ~/demo_ws/devel/setup.bash; roslaunch basic_launch sensor_init.launch; sleep 10"

echo "Starting Polaris Gem visualization..."
gnome-terminal -- sh -c "source ~/demo_ws/devel/setup.bash; roslaunch basic_launch visualization.launch; sleep 10"

echo "Starting joystick..."
gnome-terminal -- sh -c "source ~/demo_ws/devel/setup.bash; roslaunch basic_launch dbw_joystick.launch; sleep 10"
​
echo "Starting autoware docker..."
gnome-terminal -- sh -c "cd ~/autoware_ai_docker/generic; ./run.sh -t local -b ~/autoware_ai_local; source Autoware/install/setup.bash; roslaunch runtime_manager runtime_manager.launch"