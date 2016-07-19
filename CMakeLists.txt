CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project (InteractiveLearn)

#  ask to use -DCMAKE_PREFIX_PATH=/groups/scheffer/home/paragt/miniconda/envs/tst_my/

#   include_directories (/groups/scheffer/home/paragt/miniconda/envs/tst_my/include/)

   include_directories (IterativeLearners)
   include_directories (SemiSupervised)
   include_directories (Classifier)

#    link_directories (/groups/scheffer/home/paragt/miniconda/envs/tst_my/lib) 

   add_subdirectory (IterativeLearners)
   add_subdirectory (Classifier)
   add_subdirectory (SemiSupervised)

    
#    set (CMAKE_C_COMPILER "/groups/scheffer/home/paragt/miniconda/envs/test-my-install/bin/gcc")
#    set (CMAKE_CXX_COMPILER "/groups/scheffer/home/paragt/miniconda/envs/test-my-install/bin/g++")
   set(CMAKE_CXX_FLAGS "-std=c++0x")
#    set(CMAKE_CXX_STANDARD 11)
#    set(CMAKE_CXX_STANDARD_REQUIRED ON)


#    add_executable (predict_pixels predict_pixels.cpp pixel_detector.cpp) 
#    add_executable (test_vigra_watershed test_vigra_watershed.cpp) 


#    add_library(Classifier SHARED Classifier/opencvRFclassifier.cpp
   add_executable (InteractiveLearnMain Interactive_learn_main.cpp Dataset.cpp)  
#    add_library(SemiSupervised SHARED SemiSupervised/kmeans.cpp SemiSupervised/weightmatrix1.cpp)

   target_link_libraries (InteractiveLearnMain IterativeLearners Classifier SemiSupervised vigraimpex opencv_ml opencv_core boost_python boost_thread boost_system boost_chrono python2.7 png hdf5 hdf5_hl)  
#    target_link_libraries (predict_pixels Classifier SemiSupervised Utilities vigraimpex opencv_ml opencv_core boost_thread boost_system boost_chrono hdf5 hdf5_hl)  
#    target_link_libraries (test_vigra_watershed Utilities vigraimpex opencv_ml opencv_core boost_thread boost_system boost_chrono hdf5 hdf5_hl)  


############################################################################
# endif()
