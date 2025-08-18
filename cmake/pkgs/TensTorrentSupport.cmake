option(USE_TENSTORRENT "Enable support for TensTorrent")

if (USE_TENSTORRENT)
    # Look for TT-Metalium installation
    find_path(TT_METAL_INCLUDE_DIR
        NAMES tt-metalium/host_api.hpp
        HINTS
            ENV TT_METAL_HOME
            /opt/tt-metal
        PATH_SUFFIXES include build_Release/include
    )
    
    find_library(TT_METAL_LIB
        NAMES tt_metal
        HINTS
            ENV TT_METAL_HOME
            /opt/tt-metal
        PATH_SUFFIXES lib build/lib build_Release/lib
    )
    
    if (TT_METAL_INCLUDE_DIR AND TT_METAL_LIB)
        message(STATUS "Found TT-Metalium: ${TT_METAL_LIB}")
        message(STATUS "TT-Metalium include: ${TT_METAL_INCLUDE_DIR}")
        
        # Add include directories
        include_directories(${TT_METAL_INCLUDE_DIR})
        
        # Add to common link libraries
        set(COMMON_LINK_LIBRARIES ${COMMON_LINK_LIBRARIES} ${TT_METAL_LIB})
        
        # Define preprocessor macro
        add_definitions(-DUSE_TENSTORRENT)
        
        # Set TT-specific compile flags if needed
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
    else()
        message(FATAL_ERROR "TT-Metalium not found. Please set TT_METAL_HOME environment variable or install TT-Metalium")
    endif()
endif()