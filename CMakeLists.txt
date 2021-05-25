cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torchscript_example)

find_package(Torch REQUIRED)
find_package( OpenCV REQUIRED )
include_directories("/usr/local/include")
# Comment following line if you dont need cnpy
LINK_DIRECTORIES("/home/soubarna/Softwares/cnpy/build")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(run_torchscript_module run_torchscript_module.cpp)
target_link_libraries(run_torchscript_module "${TORCH_LIBRARIES}" "${OpenCV_LIBS}" cnpy)
set_property(TARGET run_torchscript_module PROPERTY CXX_STANDARD 14)



