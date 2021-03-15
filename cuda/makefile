DEBUG = 0

SOURCE_CU_LIST = \
needle.cu

EXE_LIST = $(SOURCE_CU_LIST:.cu=.exe)

CC = nvcc -Xcompiler "/openmp"
CC_FLAGS = -Xptxas -v -arch=sm_61 -prec-sqrt true
ifeq ($(DEBUG), 1)
CC_FLAGS += -Og -DDEBUG
else
CC_FLAGS += -O2
endif


all: $(EXE_LIST) | makefile

%.exe : %.cu | makefile
	$(CC) $(CC_FLAGS) -maxrregcount 32   -o $(@) $(<)  &&  rm *.lib *.exp


clear:
	rm -f *.exe *.lib *.exp  *.out*.txt; \
	clear

