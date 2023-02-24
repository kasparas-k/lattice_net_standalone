#cmake 3.12 seems to have a problem finisng BLAS
cmake_minimum_required(VERSION 3.13...3.22)

project(lattice_net)


###   VARIABLES   ##############################################################
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -O3")

set(CMAKE_CXX_STANDARD 17) #we need c++17 because this solves alignment issues with eigen http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1409
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


####   GLOBAL OPTIONS   ###https://stackoverflow.com/questions/15201064/cmake-conditional-preprocessor-define-on-code

######   PACKAGES   ############################################################
# find_package(catkin REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs calib3d)
find_package(CUDA REQUIRED)

# get and append paths for finding dep
execute_process( #do it like this https://github.com/facebookresearch/hanabi_SAD/blob/6e4ed590f5912fcb99633f4c224778a3ba78879b/rela/CMakeLists.txt#L10
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TorchPath
)
list(APPEND CMAKE_PREFIX_PATH ${TorchPath})
find_package(Torch REQUIRED)

set(pybind11_DIR ${TorchPath}/include/pybind11)
message(STATUS "TORCHPATH FOR PYBIND: ${TorchPath}")
find_package(pybind11 REQUIRED)

###   INCLUDES   #########################################################
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/deps)
include_directories(${CMAKE_SOURCE_DIR}/deps/include)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${DATALOADERS_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

###   SET ALL THE GLOBAL OPTIONS #########################################

#pybind
pybind11_add_module(latticenet ${PROJECT_SOURCE_DIR}/src/PyBridge.cxx )

#lib
# add_library( latticenet_cpp SHARED ${MY_SRC}  ${DEPS_SRC}  )

##    CUDA library  ######################
#configure the include paths of jitify so that the kernels can properly include other ones
configure_file(${PROJECT_SOURCE_DIR}/include/lattice_net/jitify_helper/jitify_options.hpp.in ${PROJECT_SOURCE_DIR}/include/lattice_net/jitify_helper/jitify_options.hpp)
install(FILES "${PROJECT_SOURCE_DIR}/include/lattice_net/jitify_helper/jitify_options.hpp" DESTINATION ${PROJECT_SOURCE_DIR}/include/lattice_net/jitify_helper/)

cuda_add_library(latticenet_cu  SHARED  ${CMAKE_SOURCE_DIR}/src/Lattice.cu
                                        ${CMAKE_SOURCE_DIR}/src/HashTable.cu
                                        )

###   LIBS   ###############################################
if(${TORCH_FOUND})
    set(LIBS ${LIBS} ${TORCH_LIBRARIES} )
    #torch 1.5.0 and above mess with pybind and we therefore need to link against libtorch_python.so also
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
    if(TORCH_PYTHON_LIBRARY)
        message(STATUS "Linking to torch_python_library")
        set(LIBS ${LIBS} ${TORCH_PYTHON_LIBRARY} )
    endif()
endif()
# set(LIBS ${LIBS} ${catkin_LIBRARIES} ${EASYPBR_LIBRARY} ${DATALOADERS_LIBRARY} ${CUDA_LIBRARIES} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set(LIBS ${LIBS} ${catkin_LIBRARIES} ${DATALOADERS_LIBRARY} ${CUDA_LIBRARIES} ${TORCH_LIBRARIES} ${OpenCV_LIBS})

# message(STATUS "LIBS: ${LIBS}")

target_link_libraries(latticenet_cu ${LIBS})
target_link_libraries(latticenet PRIVATE latticenet_cu )

#definitions for cmake variables that are necesarry during runtime
target_compile_definitions(latticenet_cu PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}") #point to the cmakelist folder of the easy_pbr
target_compile_definitions(latticenet_cu PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}") # points to the CMakeList folder of whichever project included easy_pbr
