#!/bin/sh
# nm --defined-only -g pyflexlib.so
g++ -shared -Wl,-soname,pyflexlib  \
    ../external/glad/egl.c \
    ../external/glad/gl.c \
    pyflex.cpp \
    ../core/core.cpp \
    ../core/aabbtree.cpp \
    ../core/extrude.cpp \
    ../core/maths.cpp \
    ../core/mesh.cpp \
    ../core/perlin.cpp \
    ../core/pfm.cpp \
    ../core/platform.cpp \
    ../core/png.cpp \
    ../core/sdf.cpp \
    ../core/tga.cpp \
    ../core/voxelize.cpp \
    ../lib/linux64/NvFlexDeviceRelease_x64.a \
    ../lib/linux64/NvFlexReleaseCUDA_x64.a \
    ../lib/linux64/NvFlexExtReleaseCUDA_x64.a \
    -I . \
    -I ../include \
    -I ../external/glad/ \
    -I ../ \
    -I ../core/ \
    -L/usr/local/cuda/lib64 \
    -lcuda -lcudart \
    -lpthread -ldl -o pyflexlib.so -fpermissive -fPIC -w
