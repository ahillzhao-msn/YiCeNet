cmake_minimum_required(VERSION 3.15)
project(yicenet_hook CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(yicenet-hook yicenet_hook.cpp)

if(WIN32)
  target_link_libraries(yicenet-hook PRIVATE ws2_32)
  target_compile_definitions(yicenet-hook PRIVATE _CRT_SECURE_NO_WARNINGS UNICODE _UNICODE)
endif()
if(MSVC)
  # Static CRT: the binary runs on machines without the VC++ redistributable.
  set_property(TARGET yicenet-hook PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
  target_compile_options(yicenet-hook PRIVATE /W4 /utf-8)
else()
  target_compile_options(yicenet-hook PRIVATE -Wall -Wextra -O2)
endif()

install(TARGETS yicenet-hook RUNTIME DESTINATION bin)
