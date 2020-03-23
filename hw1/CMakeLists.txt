cmake_minimum_required(VERSION 3.15)
project(hw1)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(
    Makefile
    MatAddArb.cu
        vecAdd.cu.bak
        vecAdd.cu)
