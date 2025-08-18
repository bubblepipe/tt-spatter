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
        # Add metalium-thirdparty to include path for reflect header
        include_directories(${TT_METAL_INCLUDE_DIR}/metalium-thirdparty)
        
        # Add to common link libraries
        set(COMMON_LINK_LIBRARIES ${COMMON_LINK_LIBRARIES} ${TT_METAL_LIB})
        
        # Define preprocessor macro
        add_definitions(-DUSE_TENSTORRENT)
        
        # Tell spdlog to use external fmt library instead of bundled headers
        add_definitions(-DSPDLOG_FMT_EXTERNAL=1)
        
        # Find and link fmt library
        find_library(FMT_LIB
            NAMES fmt
            HINTS
                ENV TT_METAL_HOME
                ${TT_METAL_INCLUDE_DIR}/../lib
            PATH_SUFFIXES lib build_Release/lib
        )
        
        # Find and link spdlog library
        find_library(SPDLOG_LIB
            NAMES spdlog
            HINTS
                ENV TT_METAL_HOME
                ${TT_METAL_INCLUDE_DIR}/../lib
            PATH_SUFFIXES lib build_Release/lib
        )
        
        # Set TT_METAL_LIBS for tenstorrent_backend library
        set(TT_METAL_LIBS ${TT_METAL_LIB})
        
        # Add to link libraries
        if(FMT_LIB)
            set(COMMON_LINK_LIBRARIES ${COMMON_LINK_LIBRARIES} ${FMT_LIB})
            set(TT_METAL_LIBS ${TT_METAL_LIBS} ${FMT_LIB})
            message(STATUS "Found fmt library: ${FMT_LIB}")
        endif()
        if(SPDLOG_LIB)
            set(COMMON_LINK_LIBRARIES ${COMMON_LINK_LIBRARIES} ${SPDLOG_LIB})
            set(TT_METAL_LIBS ${TT_METAL_LIBS} ${SPDLOG_LIB})
            message(STATUS "Found spdlog library: ${SPDLOG_LIB}")
        endif()
        
        # C++20 is now set per-target for tenstorrent_backend
    else()
        message(FATAL_ERROR "TT-Metalium not found. Please set TT_METAL_HOME environment variable or install TT-Metalium")
    endif()
endif()