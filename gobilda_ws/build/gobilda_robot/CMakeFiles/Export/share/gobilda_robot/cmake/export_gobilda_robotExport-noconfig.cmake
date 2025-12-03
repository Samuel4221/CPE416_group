#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gobilda_robot::gobilda_robot" for configuration ""
set_property(TARGET gobilda_robot::gobilda_robot APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(gobilda_robot::gobilda_robot PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libgobilda_robot.so"
  IMPORTED_SONAME_NOCONFIG "libgobilda_robot.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS gobilda_robot::gobilda_robot )
list(APPEND _IMPORT_CHECK_FILES_FOR_gobilda_robot::gobilda_robot "${_IMPORT_PREFIX}/lib/libgobilda_robot.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
