find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED
)
include(UseCython)

cython_transpile(
  core.pyx
  LANGUAGE C
  OUTPUT_VARIABLE core_c
)
python_add_library(core MODULE ${core_c} WITH_SOABI)

if(WIN32)
  target_compile_definitions(core PRIVATE __WIN32__=1)
endif()
if(MSVC)
  target_compile_options(core PRIVATE /std:c11)
endif()

target_include_directories(core PRIVATE ${CMAKE_SOURCE_DIR}/cryosparc/include)
target_sources(
  core
  PUBLIC
    core.pyx
    ${CMAKE_SOURCE_DIR}/cryosparc/include/lz4/lib/lz4.h
    ${CMAKE_SOURCE_DIR}/cryosparc/include/lz4/lib/lz4.c
    ${CMAKE_SOURCE_DIR}/cryosparc/include/cryosparc-tools/dataset.h
    lz4.pxd
    dataset.pxd
    dataset.c
)

install(TARGETS core DESTINATION cryosparc/dataset)
