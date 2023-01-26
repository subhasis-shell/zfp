SET(CMAKE_C_FLAGS "-D__HIP_PLATFORM_AMD__")
SET(CMAKE_CXX_FLAGS "-D__HIP_PLATFORM_AMD__")

add_executable(zfpcmd zfp.c)

# protect against LNK1114: cannot overwrite the original file 'lib/Release/zfp.lib'; error code 32;
# rationale: linker can't handle the case of an executable file having the same name as a library file
if(NOT MSVC)
  set_property(TARGET zfpcmd PROPERTY OUTPUT_NAME zfp)
endif()

target_link_libraries(zfpcmd zfp)
if(HAVE_LIBM_MATH)
  target_link_libraries(zfpcmd m)
endif()

if(BUILD_UTILITIES)
  install(TARGETS zfpcmd
    DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()
