# Performance testing
These rough tests give an idea about which components will hold up a real-time
processing chain and which components are good targets for optimization. Only
frame processing components are tested because they are orders of magnitude
slower than position processing components.

## Machine
Custom Desktop<br />
Intel Core i7-5820K CPU @ 3.30GHz<br />
GeForce GTX 970 GPU with CUDA 7.5

### Task
1000x 1MP frames served using `oat-frameserve test` which has
the following runtime characteristics with no listening
components:
- real	0m0.073s
- user	0m0.049s
- sys	0m0.026s

### Results

#### oat-framefilt

- `bsub`
  - real	0m0.881s
  - user	0m0.079s
  - sys	    0m0.037s

- `mask`
  - real	0m1.300s
  - user	0m0.074s
  - sys	    0m0.041s

- `mog`
  - real	0m1.744s
  - user	0m0.077s
  - sys	    0m0.039s

- `undistort`
  - real	0m29.946s
  - user    0m0.066s
  - sys	    0m0.068s
  - Note: slower than laptop...
  - Note: replace open-cv implementation with shader. There are lots of
    tutorials on this.


#### oat-posidet

- `diff`
  - real	0m2.835s
  - user	0m0.064s
  - sys	    0m0.056s

- `hsv`
  - real	0m4.683s
  - user	0m0.071s
  - sys	    0m0.067s

- `trsh`
  - real	0m0.545s
  - user	0m0.028s
  - sys	    0m0.012s

## Machine
Lenovo ThinkPad X1 Carbon 3rdi<br />
Intel Core i7-5600U CPU @ 2.60GHzi

### Task
1000x 1MP frames served using `oat-frameserve test` which has the following
runtime characteristics with no listening components:

- real   0m0.051s
- user   0m0.036s
- sys    0m0.008s

### Results

#### oat-framefilt

- `bsub`
  - real    0m1.278s
  - user    0m0.080s
  - sys     0m0.008s

- `mask`
  - real    0m2.049s
  - user    0m0.060s
  - sys     0m0.020s

- `mog`
  - real    0m13.213s
  - user    0m0.056s
  - sys     0m0.068s
  - Note: no CUDA support

- `undistort`
  - real    0m24.106s
  - user    0m0.096s
  - sys     0m0.060s

#### oat-decorate

- No position source, just date and sample added to frame.
  - real    0m0.934s
  - user	0m0.064s
  - sys	    0m0.024s

- Single position source with all display options turned on.
  - real	0m6.274s
  - user	0m0.132s
  - sys	    0m0.040s

#### oat-posidet

- `diff`
  - real  0m3.701s
  - user  0m0.048s
  - sys   0m0.024s

- `hsv`
  - real  0m7.422s
  - user  0m0.064s
  - sys   0m0.028s


