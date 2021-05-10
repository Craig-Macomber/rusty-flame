# rusty-flame

GPU accelerated fractal flame generator written in rust.

Goal is to get high quality 4k 60fps interactive rendering of as many fractal flames as possible. Difficulty varies wildly depending on the fractal, but there are several tricks that can enable a wide variety to be rendered well. Currently, just affine transforms are supported with density based coloring. Other kinds of transforms and coloring will be explored in the future.

Planned rendering algorithm is:

1. Compute bonds of fractal
2. For successively higher resolutions buffers:
   - Render entire fractal iteration into buffer as tri-mesh textured with previous iteration (tri-mesh can incorporate multiple iterations in a single draw call)
3. Generate final image as full screen quad with shader does the following for each pixel:
   - Walk table of inverses of transformations, summing samples from fractal buffer
   - Applies tone-mapping to sum

Rendering is using Wgpu.

Features I'd like to implement:

- [x] GPU accelerated rendering (with Wgpu)
- [x] Bounds Computation
- [x] Render multiple iterations in a single draw call
      [x] Using generated meshes
      [x] Using instanced rendering
- [ ] Scale factor adjusted density for uniform scale factors
- [ ] Scale factor adjusted density for non-uniform scale factors
- [x] Recursive render to texture for improved quality and performance
  - [ ] Use mip-maps/summed area tables (maybe anisotropic) for efficient sampling for fractals with highly variable scale factors
- [x] Logarithmic density visualization
  - [x] Output tone-mapping
  - [x] Floating point internal buffers
  - [ ] Mitigation for overflow/saturation (normalizing and/or value custom encoding)
- [ ] Path based coloring
- Support non-affine functions:
  - [ ] Continuos
  - [ ] Discontinuous
- [ ] Localized up-sampling (invert functions and sample from fractal buffer)
- [ ] Automatic adjustments of amount of iterations and resolutions at different phases to optimize performance while keeping quality.
- [ ] Web Support
- [ ] Performance testing
- [ ] Quality testing
- [ ] Fractal editing GUI
