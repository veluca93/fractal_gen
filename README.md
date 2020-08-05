# Fractal Gen
This repository contains multiple programs to generate images or videos of
fractals, with various techniques.

## Programs
### Julia Stars
This program generates a Julia Set, coloured using the smooth colouring
technique based on escape time (see eg.  [this
link](http://linas.org/art-gallery/escape/escape.html)). On the inside of the
set, some stars are randomly generated with a density that depends on the same
function that is used for defining the colour.

### Attractor
This program draws the basin of attraction of each root of a polynomial in the
complex plane. The input file defines a shape, which is used to distort the
complex plane so that the resulting fractal will resemble the input shape.

### Flame and Fractal Video
These programs generate either an image or a video using the same technique
used in [Apophysis](https://en.wikipedia.org/wiki/Apophysis_(software)).

`fractal_video` and `fractal_video_cuda` produce frames of an animation,
obtained by modifying the algorithm in Apophysis to work in 3 dimensions.
`fractal_video_cuda` is implemented using CUDA and is significantly faster.

More information on these programs will be available soon.

## Building and running
Compile the files using `make`. The binaries will be found in the `build` folder.

## Dedication
All the programs in this repository are for Iulia, a very special girl in my
life which deserves all the happiness in the world and enjoys fractals and
stars even more than I do.

Without Iulia, this repository and the programs in it would never have existed.

Thank you for making every day of my life full of happiness!
