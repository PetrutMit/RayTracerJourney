cmake_minimum_required(VERSION 3.10)

project("RealTimeRTX" LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 17)

# Add Static Libraries
link_directories(${CMAKE_SOURCE_DIR}/lib)

# Add Source Files for RT
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/RT/render.cu)
# And for Window Drawing
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/Window/main.cpp)

# Add Header File for RT
file(GLOB_RECURSE HEADERS ${CMAKE_SOURCE_DIR}/src/RT/HeaderFiles/*.cuh)
# And for Window Drawing
file(GLOB_RECURSE HEADERS ${CMAKE_SOURCE_DIR}/src/Window/HeaderFiles/*.hpp)

# Define the executable
add_executable(RealTimeRTX ${SOURCES} ${HEADERS})

# Set the CUDA architecture
set_target_properties(RealTimeRTX PROPERTIES CUDA_ARCHITECTURES 70)

# Add CMAKE_DIR with auxiliary functions
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

################################
# Packages
################################

# Find CUDA - DEPRECATED
#find_package(CUDA REQUIRED)

# Find OpenGL
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

# Find GLM
find_package(GLM REQUIRED)
message(STATUS "GLM_INCLUDE_DIR: ${GLM_INCLUDE_DIR}")

# Find GLFW
find_package(GLFW3 REQUIRED)
message(STATUS "GLFW_INCLUDE_DIR: ${GLFW_INCLUDE_DIR}")

# Add GLAD
add_library(GLAD "thirdparty/glad.c")

# Put all the libraries together
set(LIBS ${CUDA_LIBRARIES} ${OPENGL_LIBRARY} ${GLFW3_LIBRARY} ${CMAKE_DL_LIBS} GLAD)

# Include Directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include 
    ${CMAKE_SOURCE_DIR}/include/glad
    ${CMAKE_SOURCE_DIR}/include/glm
)

# Define the libraries to link
target_link_libraries(RealTimeRTX ${LIBS})

# Create virtual folder structure for VS
if(MSVC_IDE)
	# Macro to preserve source files hierarchy in the IDE
	macro(GroupSources curdir)
		file(GLOB children RELATIVE ${PROJECT_SOURCE_DIR}/${curdir} ${PROJECT_SOURCE_DIR}/${curdir}/*)

		foreach(child ${children})
			if(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/${curdir}/${child})
				GroupSources(${curdir}/${child})
			else()
				string(REPLACE "/" "\\" groupname ${curdir})
				string(REPLACE "src" "Sources" groupname ${groupname})
				source_group(${groupname} FILES ${PROJECT_SOURCE_DIR}/${curdir}/${child})
			endif()
		endforeach()
	endmacro()

	# Run macro
	GroupSources(src)
endif()

