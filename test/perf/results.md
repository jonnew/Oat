# Performance testing

These rough tests give an idea about which components will hold up a real-time processing chain and which components are good targets for optimization. Only frame processing components are tested because they are orders of magnitude slower than position processing components.

## Machine
Custom Desktop
Intel Core i7-5820K CPU @ 3.30GHz
GeForce GTX 970 GPU with CUDA 7.5

### Task
1000x 1MP frames served using `oat-frameserve test` which has
the following runtime characteristics with no listening
components:

- real	0m0.077s
- user	0m0.038s
- sys	0m0.041s

### Results

#### oat-framefilt

-`bsub`
  - real	0m1.047s
  - user	0m0.047s
  - sys	    0m0.064s

- `mask`
  - real	0m1.390s
  - user	0m0.039s
  - sys	    0m0.075s

- `mog`
  - real	0m2.133s
  - user	0m0.060s
  - sys	    0m0.056s

- `undistort`
  - real	0m29.946s
  - user    0m0.066s
  - sys	    0m0.068s
  - Note: slower than laptop...
  - Note: replace open-cv implementation with shader. There are lots of
    tutorials on this.

#### oat-posidet

- `diff`
  - real	0m3.983s
  - user	0m0.063s
  - sys	    0m0.062s
  - Note: slower than laptop...

- `hsv`
  - real	0m9.286s
  - user	0m0.057s
  - sys	    0m0.066s
  - Note: slower than laptop...

## Machine
Lenovo ThinkPad X1 Carbon 3rd
Intel Core i7-5600U CPU @ 2.60GHz

### Task
1000x 1MP frames served using `oat-frameserve test` which has the following
runtime characteristics with no listening components:

- real   0m0.051s
- user   0m0.036s
- sys    0m0.008s

### Results

#### oat-framefilt

- `bsub`
  - real   0m1.278s
  - user   0m0.080s
  - sys    0m0.008s

- `mask`
  - real    0m2.049s
  - user    0m0.060s
  - sys     0m0.020s

- `mog`
  - real   0m13.213s
  - user   0m0.056s
  - sys    0m0.068s

- `undistort`
  - real   0m24.106s
  - user   0m0.096s
  - sys    0m0.060s

#### oat-posidet

- `diff`
  - real  0m3.701s
  - user  0m0.048s
  - sys   0m0.024s

- `hsv`
  - real  0m7.422s
  - user  0m0.064s
  - sys   0m0.028s
