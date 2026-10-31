# Derived from radiance-exl3 (https://github.com/Dyluhn/radiance-exl3, Apache-2.0, Copyright 2026 Dylan Johnston) and radiance's libr4d (Apache-2.0); EXL3 format per exllamav3 (MIT). Modified for this project; see NOTICE.
# exl3r4d -- the gfx1201 kernel library for the dense EXL3 linear (gemm_exl3_nt / exl3_dense). Built from the
# repository's top level, or on its own against an installed radiance:
#
#   cmake -S kernels/exl3r4d -B build-exl3r4d -DCMAKE_PREFIX_PATH=<radiance install prefix>
#
# DEVICE ONLY. The library exists when the radiance found has a device build (RAD_HIP_FOUND), -DRAD_WITH_HIP is
# not OFF, and the build targets include gfx1201 (rad_plugin_gpu_targets); otherwise it is left out and said so --
# a host-only build has nothing for it to serve (exl3dense's scalar rows are its host twin).
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(exl3r4d LANGUAGES C CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
  enable_testing()
  set(EXL3R4D_STANDALONE ON)
endif()
enable_language(C)

if(NOT RADIANCE_ABI_VERSION EQUAL 31)
  message(FATAL_ERROR "exl3r4d is written against radiance ABI 31; the radiance found is ABI "
                      "${RADIANCE_ABI_VERSION}")
endif()

get_filename_component(EXL3_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
foreach(_f exl3_abi.h exl3_encoding.h exl3_decode.c exl3_layout.c)
  set(_d exl3import/common)
  if(_f MATCHES "\.h$")
    if(NOT _f STREQUAL "exl3_decode.h" AND NOT _f STREQUAL "exl3_layout.h")
      set(_d exl3import/include)
    endif()
  endif()
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../${_d}/${_f}")
    message(FATAL_ERROR "exl3r4d needs ${_d}/${_f} (plugin/exl3import)")
  endif()
endforeach()
set(EXL3R4D_C_FLAGS -ffp-contract=off -Wall -Wextra)

set(EXL3R4D_HIP OFF)
if(RAD_HIP_FOUND AND NOT (DEFINED RAD_WITH_HIP AND RAD_WITH_HIP STREQUAL "OFF"))
  rad_plugin_gpu_targets(EXL3R4D_TARGETS "^gfx1201$")
  if(EXL3R4D_TARGETS)
    set(EXL3R4D_HIP ON)
  else()
    message(STATUS "exl3r4d: left out -- this build targets '${RAD_GPU_TARGETS}', and the library is gfx1201 code")
  endif()
else()
  message(STATUS "exl3r4d: left out -- no device build (RAD_HIP_FOUND=${RAD_HIP_FOUND}, RAD_WITH_HIP=${RAD_WITH_HIP})")
endif()

if(EXL3R4D_HIP)
  if(EXL3R4D_STANDALONE AND NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${EXL3R4D_TARGETS}")
  endif()
  if(EXL3R4D_STANDALONE)
    enable_language(HIP)
  endif()
  rad_add_plugin(exl3r4d kernels
    SOURCES rows.c layout.c nt.c ../exl3import/common/exl3_decode.c ../exl3import/common/exl3_layout.c
    HIP_SOURCES exl3r4d_nt.hip
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ../exl3import/include ../exl3import/common
    GPU_TARGETS ${EXL3R4D_TARGETS})
  set_target_properties(exl3r4d PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
  # The file is exl3_r4d.so (the plugin's name stays exl3r4d): an unconfigured load orders kernel libraries by file
  # name and the first with a matching row serves the op, so this one must come before exl3dense.so to serve
  # exl3_dense by default. exl3dense keeps what this row refuses (K = 1, the mcg / 3inst codebooks).
  set_target_properties(exl3r4d PROPERTIES OUTPUT_NAME exl3_r4d)
  # -ffp-contract=off on the host code too (rad_hipify sets it for the device code): the layout code does no
  # arithmetic that could contract, and the flag keeps it that way.
  target_compile_options(exl3r4d PRIVATE $<$<COMPILE_LANGUAGE:C>:${EXL3R4D_C_FLAGS}>)
  # P9-L7 (EXL3R4D_ALIGN_PAGE): every device function starts on a 4 KB boundary. Two builds with instruction-identical
  # kernels that differ only in where the kernels landed (the gate_up GEMM at 0x43000 vs 0x58a00) measured the tp1 M=3
  # split chain at 146.0 vs 155.2 us a layer (perfmoe, 0000:03:00.0, four sessions; the gate_up GEMM itself 88 vs 101 us
  # inside the chain, traced; equal alone and at every other shape tried); the second tree with 4 KB alignment read
  # 146.0 again (sK-1). Mechanism open (not the K loop crossing a page: a build whose loop crosses one is fast). The
  # cost is padding, < 4 KB a kernel. OFF puts the kernels wherever the linker packs them.
  option(EXL3R4D_ALIGN_PAGE "start every exl3r4d device function on a 4 KB boundary" ON)
  if(EXL3R4D_ALIGN_PAGE)
    target_compile_options(exl3r4d PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-falign-functions=4096>)
  endif()
  string(REPLACE ";" " " _exl3r4d_bt "${EXL3R4D_TARGETS}")
  target_compile_definitions(exl3r4d PRIVATE "EXL3R4D_BUILD_TARGET=\"${_exl3r4d_bt}\"")
  find_library(EXL3R4D_LIBM m)   # common/exl3_decode.c's nearbyint; part of libc where absent
  if(EXL3R4D_LIBM)
    target_link_libraries(exl3r4d PRIVATE ${EXL3R4D_LIBM})
  endif()
endif()


