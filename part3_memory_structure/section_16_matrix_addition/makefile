#!/bin/make
#
# (c) 2021-2022. biztripcru@gmail.com. All rights reserved.
#

END =
SRCS = \
	$(wildcard *.cu) \
	$(wildcard *.cpp) \
	$(END)
SRCS := $(filter-out common.cpp, $(SRCS))
SRCS := $(filter-out image.cpp, $(SRCS))
SRCS := $(filter-out 04e-error-stream.cu, $(SRCS))
SRCS := $(filter-out $(wildcard *-expired-*), $(SRCS))
SRCS := $(filter-out $(wildcard 103-*), $(SRCS))
SRCS := $(filter-out $(wildcard 104-*), $(SRCS))
SRCS := $(filter-out $(wildcard 111-*), $(SRCS))
SRCS := $(filter-out $(wildcard 112-*), $(SRCS))
#SRCS := $(filter-out 29e-count-block.cu, $(SRCS) ) # SM_60 needed
#SRCS := $(filter-out 30e-hist-shared.cu, $(SRCS) ) # SM_60 needed
#SRCS := $(filter-out 30f-hist4-shared.cu, $(SRCS) ) # SM_60 needed
#SRCS := $(filter-out 31c-sum-shared.cu, $(SRCS) ) # SM_60 needed
OBJS1 = $(SRCS:.cpp=.obj)
OBJS = $(OBJS1:.cu=.obj)
EXES1 = $(SRCS:.cpp=.exe)
EXES = $(EXES1:.cu=.exe)

.SUFFIXES:	.cpp .cu .obj .exe

ifeq ($(OS),Windows_NT) # windows

VS_VER = 14.29.30133

NVCC = nvcc.exe
NVCCFLAGS += --compiler-bindir "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\$(VS_VER)\\bin\\Hostx64\\x64"
NVCCFLAGS += -O2 -Xcompiler -wd4819
#NVCCFLAGS += -arch sm_52 # failed for atomicAdd_block()
NVCCFLAGS += -arch sm_60

# NVCCFLAGS += -w -Xptxas -v -maxrregcount=2

#NVCCFLAGS += --compiler-options -Wall --linker-options -Wall

%.obj:	%.cpp
	$(NVCC) $(NVCCFLAGS) --compile -o $@ $<
%.obj:	%.cu
	$(NVCC) $(NVCCFLAGS) --compile -o $@ $<
%.exe:	%.obj
	$(NVCC) $(NVCCFLAGS) --link -o $@ $<
%.exe:	%.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ $<
%.exe:	%.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

else # linux

NVCC = nvcc
#NVCCFLAGS += -gencode=arch=compute_52,code=\"sm_52,compute_52\" -arch=sm_52 # failed for atomicAdd_block()
NVCCFLAGS += -gencode=arch=compute_60,code=\"sm_60,compute_60\" -arch=sm_60
#NVCCFLAGS += -gencode=arch=compute_75,code=\"sm_75,compute_75\" -arch=sm_75
NVCCFLAGS += -O2
#NVCCFLAGS += -Xptxas -v

CFLAGS += -std=c11 -O2 -I../include
CXXFLAGS += -std=c++11 -O2 -I../include

%.obj:	%.cu
	$(NVCC) $(NVCCFLAGS) --compile -o $@ $<
%.obj:	%.cpp
	$(COMPILE.cc) -o $@ $<
%.obj:	%.c
	$(COMPILE.c) -o $@ $<
%.exe:	%.obj
	$(LINK.cc) $^ -o $@ $(LOADLIBES) $(LDLIBS)
%.exe:	%.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

endif


# CUDA linker command line

default: $(EXES)

clean:
	/bin/rm -f *.exp *.lib
clobber:	clean
	/bin/rm -f *.exe
