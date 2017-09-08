cmake_minimum_required(VERSION 3.1.0)

project(yushiqi_facedet)

include_directories(../include)
link_directories(../lib)

if (MSVC)                                                        # - Microsoft Visual C++
    if (CMAKE_CL_64)  # -- 64-bit builds.
        set(facedet_required_libs libfacedetect-x64)
		# [...]
    else ()                                                      # -- 32-bit builds.
		set(facedet_required_libs libfacedetect)
		# [...]
    endif ()
        # [...]
else ()	
    message(FATAL_ERROR "\nError: Only Microsoft Visual Studio is currently supported.\n")
endif ()

# Build options
option(USE_OPENMP      "Set to ON to build use openmp"  ON)
option(USE_OPENMP      "Set to ON to build use openmp"  ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# Use OpenMP
if (USE_OPENMP)
    find_package(OpenMP QUIET)
    if (OPENMP_FOUND)
        message(STATUS "Use OpenMP")
        add_definitions(-D_OPENMP)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

message(STATUS "Build examples.")

find_package(OpenCV)

if (NOT OpenCV_FOUND)
	message(WARNING "OpenCV not found. Test will not be built.")
else()
	include_directories(${OpenCV_INCLUDE_DIRS})
	list(APPEND facedet_required_libs ${OpenCV_LIBS})
	
	add_executable(example libfacedetect-example.cpp)
	target_link_libraries(example ${facedet_required_libs})

	add_executable(benchmark benchmark.cpp)
	target_link_libraries(benchmark ${facedet_required_libs})
endif()

