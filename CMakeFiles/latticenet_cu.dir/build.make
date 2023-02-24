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
include CMakeFiles/latticenet_cu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/latticenet_cu.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/latticenet_cu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/latticenet_cu.dir/flags.make

CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_Lattice.cu.o: CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_Lattice.cu.o.depend
CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_Lattice.cu.o: CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_Lattice.cu.o.RelWithDebInfo.cmake
CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_Lattice.cu.o: src/Lattice.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_Lattice.cu.o"
	cd /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles/latticenet_cu.dir/src && /usr/bin/cmake -E make_directory /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles/latticenet_cu.dir/src/.
	cd /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles/latticenet_cu.dir/src && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=RelWithDebInfo -D generated_file:STRING=/home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles/latticenet_cu.dir/src/./latticenet_cu_generated_Lattice.cu.o -D generated_cubin_file:STRING=/home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles/latticenet_cu.dir/src/./latticenet_cu_generated_Lattice.cu.o.cubin.txt -P /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_Lattice.cu.o.RelWithDebInfo.cmake

CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_HashTable.cu.o: CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_HashTable.cu.o.depend
CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_HashTable.cu.o: CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_HashTable.cu.o.RelWithDebInfo.cmake
CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_HashTable.cu.o: src/HashTable.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building NVCC (Device) object CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_HashTable.cu.o"
	cd /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles/latticenet_cu.dir/src && /usr/bin/cmake -E make_directory /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles/latticenet_cu.dir/src/.
	cd /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles/latticenet_cu.dir/src && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=RelWithDebInfo -D generated_file:STRING=/home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles/latticenet_cu.dir/src/./latticenet_cu_generated_HashTable.cu.o -D generated_cubin_file:STRING=/home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles/latticenet_cu.dir/src/./latticenet_cu_generated_HashTable.cu.o.cubin.txt -P /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_HashTable.cu.o.RelWithDebInfo.cmake

# Object files for target latticenet_cu
latticenet_cu_OBJECTS =

# External object files for target latticenet_cu
latticenet_cu_EXTERNAL_OBJECTS = \
"/home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_Lattice.cu.o" \
"/home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_HashTable.cu.o"

liblatticenet_cu.so: CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_Lattice.cu.o
liblatticenet_cu.so: CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_HashTable.cu.o
liblatticenet_cu.so: CMakeFiles/latticenet_cu.dir/build.make
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/libcudart.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/python3.10/site-packages/torch/lib/libc10.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/stubs/libcuda.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/libnvrtc.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/libnvToolsExt.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/libcudart.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/python3.10/site-packages/torch/lib/libc10_cuda.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/python3.10/site-packages/torch/lib/libtorch_python.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/libcudart.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/python3.10/site-packages/torch/lib/libtorch.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/python3.10/site-packages/torch/lib/libc10.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/stubs/libcuda.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/libnvrtc.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/libnvToolsExt.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/libcudart.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/python3.10/site-packages/torch/lib/libc10_cuda.so
liblatticenet_cu.so: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.5.4d
liblatticenet_cu.so: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.5.4d
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/python3.10/site-packages/torch/lib/libc10_cuda.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/python3.10/site-packages/torch/lib/libc10.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/libcufft.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/libcurand.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/libcublas.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/libcudnn.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/python3.10/site-packages/torch/lib/libtorch_python.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/libcudart.so
liblatticenet_cu.so: /home/kasparas/miniconda3/envs/torch/lib/libnvToolsExt.so
liblatticenet_cu.so: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.5.4d
liblatticenet_cu.so: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.5.4d
liblatticenet_cu.so: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.5.4d
liblatticenet_cu.so: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.5.4d
liblatticenet_cu.so: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.5.4d
liblatticenet_cu.so: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.5.4d
liblatticenet_cu.so: CMakeFiles/latticenet_cu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library liblatticenet_cu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/latticenet_cu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/latticenet_cu.dir/build: liblatticenet_cu.so
.PHONY : CMakeFiles/latticenet_cu.dir/build

CMakeFiles/latticenet_cu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/latticenet_cu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/latticenet_cu.dir/clean

CMakeFiles/latticenet_cu.dir/depend: CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_HashTable.cu.o
CMakeFiles/latticenet_cu.dir/depend: CMakeFiles/latticenet_cu.dir/src/latticenet_cu_generated_Lattice.cu.o
	cd /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone /home/kasparas/Documents/pointcloud_cnn/algorithms/myfork/lattice_net_standalone/CMakeFiles/latticenet_cu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/latticenet_cu.dir/depend

