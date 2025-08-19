cmake_minimum_required(VERSION 3.13)

project(lattice_net)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

execute_process( #do it like this https://github.com/facebookresearch/hanabi_SAD/blob/6e4ed590f5912fcb99633f4c224778a3ba78879b/rela/CMakeLists.txt#L10
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TorchPath
)
list(APPEND CMAKE_PREFIX_PATH ${TorchPath})
find_package(Torch REQUIRED)

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/deps
    ${CMAKE_SOURCE_DIR}/deps/include
    ${EIGEN3_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

pybind11_add_module(latticenet ${PROJECT_SOURCE_DIR}/src/PyBridge.cxx )

configure_file(${PROJECT_SOURCE_DIR}/include/lattice_net/jitify_helper/jitify_options.hpp.in ${PROJECT_SOURCE_DIR}/include/lattice_net/jitify_helper/jitify_options.hpp)
install(FILES "${PROJECT_SOURCE_DIR}/include/lattice_net/jitify_helper/jitify_options.hpp" DESTINATION ${PROJECT_SOURCE_DIR}/include/lattice_net/jitify_helper/)

add_library(latticenet_cu  SHARED
    ${CMAKE_SOURCE_DIR}/src/Lattice.cu
    ${CMAKE_SOURCE_DIR}/src/HashTable.cu  
    # ${CMAKE_SOURCE_DIR}/src/ModelParams.cxx
    # ${CMAKE_SOURCE_DIR}/src/TrainParams.cxx
    # ${CMAKE_SOURCE_DIR}/src/EvalParams.cxx
)

set(LIBS
    ${LIBS}
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    ${OpenCV_LIBS}
    CUDA::cuda_driver
)

set_target_properties(latticenet_cu PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/latticenet_py/backend/
)
set_target_properties(latticenet PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/latticenet_py/backend/
)

target_link_libraries(latticenet_cu PUBLIC ${LIBS} )
target_link_libraries(latticenet PRIVATE latticenet_cu pybind11::module)

# Hardcodes path to this repository into some of the code
target_compile_definitions(latticenet_cu PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_compile_definitions(latticenet_cu PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
