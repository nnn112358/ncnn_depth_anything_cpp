# Compiler settings - Change the compiler to g++ and set the C++ standard to C++14
CC = g++
CFLAGS = -std=c++14 $(shell pkg-config --cflags opencv4)

# Linker flags - Use pkg-config to get the flags needed to link with OpenCV 4
LDFLAGS = $(shell pkg-config --libs opencv4)

# Include paths - Add the NCNN include directories
INCLUDE = -I/opt/external/ncnn/install/x64/include/ncnn

# Library paths and libraries - Specify where to find the NCNN library and link against it, enable OpenMP
LIB = -L/opt/external/ncnn/install/x64/lib/ -lncnn -fopenmp

# Target executable name
target = ncnn_main

# Source files
sources = ncnn_main.cpp

# Object files
objects = $(addprefix obj/, $(sources:.cpp=.o))

# Link the program
$(target): $(objects)
	$(CC) -o $@ $^ $(LIB) $(LDFLAGS) $(CFLAGS)

# Compile the source files into object files
obj/%.o: %.cpp
	@[ -d obj ] || mkdir -p obj  # Ensure the obj directory exists
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

# Default target
all: $(target)

# Clean up
clean:
	rm -f $(objects) $(target)

