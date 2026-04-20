cmake_minimum_required(VERSION 3.10.2)

project(piper_description)

# Default ot C++14
if(NOT CMAKE_CXX_STANDARD)
	set(CMAKE_CXX_STANDARD 14)
endif()
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
ament_package()

#find_package(roslaunch)

foreach(dir urdf rviz launch meshes config)

	install(DIRECTORY ${dir}/
		DESTINATION share/${PROJECT_NAME}/${dir})
endforeach(dir)
# install(DIRECTORY meshes urdf DESTINATION share/${PROJECT_NAME})
