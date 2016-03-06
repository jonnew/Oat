## Performance testing

### Machine
Lenovo ThinkPad X1 Carbon 3rd
Intel Core i7-5600U CPU @ 2.60GHz

### Task
1000x 1MP frames served using `oat-frameserve test` which has the following
runtime characteristics with no listening components:

- real   0m0.051s
- user   0m0.036s
- sys    0m0.008s

## Results

### oat-framefilt

#### `bsub`
- real   0m1.278s
- user   0m0.080s
- sys    0m0.008s

#### `mask`
- real    0m2.049s
- user    0m0.060s
- sys     0m0.020s

#### `mog`
- real   0m13.213s
- user   0m0.056s
- sys    0m0.068s

#### `undistort`
- real   0m24.106s
- user   0m0.096s
- sys    0m0.060s

### oat-posidet

#### `diff`
- real  0m3.701s
- user  0m0.048s
- sys   0m0.024s

#### `hsv`
- real  0m7.422s
- user  0m0.064s
- sys   0m0.028s

### oat-posifilt

#### `kalman`
real    0m0.067s
user    0m0.040s
sys     0m0.012s

