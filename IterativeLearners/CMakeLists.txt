# CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project (IterativeLearners)

# IF(NOT CMAKE_BUILD_TYPE)
#     SET (CMAKE_BUILD_TYPE Release)
# ENDIF(NOT CMAKE_BUILD_TYPE)
# 
# SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
# SET(CMAKE_CXX_FLAGS_DEBUG "-ggdb")
# SET(CMAKE_CXX_LINK_FLAGS "-O3")
# SET(CMAKE_DEBUG_POSTFIX "-g")

set (SOURCES IterativeLearn.cpp IterativeLearn_semi.cpp IterativeLearn_unc.cpp IterativeLearn_cotrain.cpp)

add_library (IterativeLearners SHARED ${SOURCES})
# 
# if (NOT BUILDEM_DIR STREQUAL "None")
#     add_dependencies (Classifier ${vigra_NAME} ${opencv_NAME})
# endif()
