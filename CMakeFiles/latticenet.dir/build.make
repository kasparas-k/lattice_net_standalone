# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone

# Include any dependencies generated for this target.
include CMakeFiles/latticenet.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/latticenet.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/latticenet.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/latticenet.dir/flags.make

CMakeFiles/latticenet.dir/src/PyBridge.cxx.o: CMakeFiles/latticenet.dir/flags.make
CMakeFiles/latticenet.dir/src/PyBridge.cxx.o: src/PyBridge.cxx
CMakeFiles/latticenet.dir/src/PyBridge.cxx.o: CMakeFiles/latticenet.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/latticenet.dir/src/PyBridge.cxx.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/latticenet.dir/src/PyBridge.cxx.o -MF CMakeFiles/latticenet.dir/src/PyBridge.cxx.o.d -o CMakeFiles/latticenet.dir/src/PyBridge.cxx.o -c /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/src/PyBridge.cxx

CMakeFiles/latticenet.dir/src/PyBridge.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/latticenet.dir/src/PyBridge.cxx.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/src/PyBridge.cxx > CMakeFiles/latticenet.dir/src/PyBridge.cxx.i

CMakeFiles/latticenet.dir/src/PyBridge.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/latticenet.dir/src/PyBridge.cxx.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/src/PyBridge.cxx -o CMakeFiles/latticenet.dir/src/PyBridge.cxx.s

# Object files for target latticenet
latticenet_OBJECTS = \
"CMakeFiles/latticenet.dir/src/PyBridge.cxx.o"

# External object files for target latticenet
latticenet_EXTERNAL_OBJECTS =

latticenet.cpython-310-x86_64-linux-gnu.so: CMakeFiles/latticenet.dir/src/PyBridge.cxx.o
latticenet.cpython-310-x86_64-linux-gnu.so: CMakeFiles/latticenet.dir/build.make
latticenet.cpython-310-x86_64-linux-gnu.so: liblatticenet_cu.so
latticenet.cpython-310-x86_64-linux-gnu.so: /home/kasparas/miniconda3/envs/torch/lib/python3.10/site-packages/torch/lib/libtorch.so
latticenet.cpython-310-x86_64-linux-gnu.so: /home/kasparas/miniconda3/envs/torch/lib/python3.10/site-packages/torch/lib/libc10_cuda.so
latticenet.cpython-310-x86_64-linux-gnu.so: /home/kasparas/miniconda3/envs/torch/lib/python3.10/site-packages/torch/lib/libc10.so
latticenet.cpython-310-x86_64-linux-gnu.so: /home/kasparas/miniconda3/envs/torch/lib/libcufft.so
latticenet.cpython-310-x86_64-linux-gnu.so: /home/kasparas/miniconda3/envs/torch/lib/libcurand.so
latticenet.cpython-310-x86_64-linux-gnu.so: /home/kasparas/miniconda3/envs/torch/lib/libcublas.so
latticenet.cpython-310-x86_64-linux-gnu.so: /home/kasparas/miniconda3/envs/torch/lib/libcudnn.so
latticenet.cpython-310-x86_64-linux-gnu.so: /home/kasparas/miniconda3/envs/torch/lib/python3.10/site-packages/torch/lib/libc10.so
latticenet.cpython-310-x86_64-linux-gnu.so: /home/kasparas/miniconda3/envs/torch/lib/stubs/libcuda.so
latticenet.cpython-310-x86_64-linux-gnu.so: /home/kasparas/miniconda3/envs/torch/lib/libnvrtc.so
latticenet.cpython-310-x86_64-linux-gnu.so: /home/kasparas/miniconda3/envs/torch/lib/python3.10/site-packages/torch/lib/libc10_cuda.so
latticenet.cpython-310-x86_64-linux-gnu.so: /home/kasparas/miniconda3/envs/torch/lib/python3.10/site-packages/torch/lib/libtorch_python.so
latticenet.cpython-310-x86_64-linux-gnu.so: /home/kasparas/miniconda3/envs/torch/lib/python3.10/site-packages/torch/lib/libc10.so
latticenet.cpython-310-x86_64-linux-gnu.so: /home/kasparas/miniconda3/envs/torch/lib/stubs/libcuda.so
latticenet.cpython-310-x86_64-linux-gnu.so: /home/kasparas/miniconda3/envs/torch/lib/libnvrtc.so
latticenet.cpython-310-x86_64-linux-gnu.so: /home/kasparas/miniconda3/envs/torch/lib/python3.10/site-packages/torch/lib/libc10_cuda.so
latticenet.cpython-310-x86_64-linux-gnu.so: /home/kasparas/miniconda3/envs/torch/lib/python3.10/site-packages/torch/lib/libtorch_python.so
latticenet.cpython-310-x86_64-linux-gnu.so: /home/kasparas/miniconda3/envs/torch/lib/libcudart.so
latticenet.cpython-310-x86_64-linux-gnu.so: /home/kasparas/miniconda3/envs/torch/lib/libnvToolsExt.so
latticenet.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.5.4d
latticenet.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.5.4d
latticenet.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.5.4d
latticenet.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.5.4d
latticenet.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.5.4d
latticenet.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.5.4d
latticenet.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.5.4d
latticenet.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.5.4d
latticenet.cpython-310-x86_64-linux-gnu.so: CMakeFiles/latticenet.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module latticenet.cpython-310-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/latticenet.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/latticenet.dir/build: latticenet.cpython-310-x86_64-linux-gnu.so
.PHONY : CMakeFiles/latticenet.dir/build

CMakeFiles/latticenet.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/latticenet.dir/cmake_clean.cmake
.PHONY : CMakeFiles/latticenet.dir/clean

CMakeFiles/latticenet.dir/depend:
	cd /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles/latticenet.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/latticenet.dir/depend

