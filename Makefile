CC=clang++
NVCC=nvcc

all: build/julia_stars build/attractor build/fractal_video build/flame build/fractal_video_cuda

build/%: %.cc lodepng/lodepng.cpp lodepng/lodepng.h common.h
	${CC} -fopenmp -latomic -I lodepng $< -o $@ -Wall -O3 -ffast-math lodepng/lodepng.cpp


build/%: %.cu lodepng/lodepng.cpp lodepng/lodepng.h common.h
	${NVCC} $< -I lodepng -o $@ -O3 lodepng/lodepng.cpp

clean:
	rm -f build/*
