cmake_minimum_required(VERSION 3.26)
project(projecteur_scene_autonome
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenCV REQUIRED)

add_executable(projecteur_scene_autonome src/main.cpp)
target_include_directories(projecteur_scene_autonome PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(projecteur_scene_autonome PRIVATE ${OpenCV_LIBS})
target_compile_definitions(projecteur_scene_autonome PRIVATE ENABLE_FAST_MATH CUDA_FAST_MATH)
