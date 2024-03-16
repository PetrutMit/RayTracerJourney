cmake_minimum_required(VERSION 3.10)

project("RealTimeRTX" LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 17)

# Add Static Libraries
link_directories(${CMAKE_SOURCE_DIR}/lib)

# Add Source Files
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/main.cpp ${CMAKE_SOURCE_DIR}/src/render.cu)

# Add Header Files
file(GLOB_RECURSE HEADERS ${CMAKE_SOURCE_DIR}/src/HeaderFiles/*.cuh ${CMAKE_SOURCE_DIR}/src/HeaderFiles/*.hpp)

# Copy Shaders to build directory
file(COPY ${CMAKE_SOURCE_DIR}/src/Shaders DESTINATION .)

# Define the executable
add_executable(RealTimeRTX ${SOURCES} ${HEADERS})

# Set RealTimeRTX as default startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT RealTimeRTX)

# Set the CUDA architecture
set_target_properties(RealTimeRTX PROPERTIES CUDA_ARCHITECTURES 70)

# Add CMAKE_DIR with auxiliary functions
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

################################
# Packages
################################

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

include(CMakePrintHelpers)

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


function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
	get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
	if(NOT "${old_flags}" STREQUAL "")
		string(REPLACE ";" "," CUDA_flags "${old_flags}")
		set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
			"$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
			)
	endif()
endfunction()

CUDA_CONVERT_FLAGS(${PROJECT_NAME})


