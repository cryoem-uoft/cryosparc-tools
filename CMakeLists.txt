cmake_minimum_required(VERSION 3.15...4.0)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/cryosparc/dataset")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cryosparc/include/cryosparc-tools/dataset.h"
    "${SKBUILD_HEADERS_DIR}/cryosparc-tools/dataset.h"
    COPYONLY
)
