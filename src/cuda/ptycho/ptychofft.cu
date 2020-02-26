#include "ptychofft.cuh"
#include "kernels.cu"

#include <stdio.h>

// constructor, memory allocation
ptychofft::ptychofft(size_t ntheta, size_t nz, size_t n, size_t nscan,
  size_t detector_shape, size_t probe_shape
) :
  ntheta(ntheta), nz(nz), n(n), nscan(nscan), detector_shape(detector_shape),
  probe_shape(probe_shape)
{
	// create batched 2D FFT plan on GPU with sizes (detector_shape, detector_shape)
  // transform shape MUST be less than or equal to input and ouput shapes.
	int ffts[2] = {(int)detector_shape, (int)detector_shape};
	cufftPlanMany(&plan2d, 2,
    ffts,                 // transform shape
    ffts, 1, detector_shape * detector_shape, // input shape
    ffts, 1, detector_shape * detector_shape, // output shape
    CUFFT_C2C,
    ntheta * nscan        // Number of FFTs to do simultaneously
  );
  // create a place to put the FFT and IFFT output.
  cudaMalloc((void**)&fft_out, ntheta * nscan * detector_shape * detector_shape * sizeof(float2));

	// init 3d thread block on GPU
	BS3d.x = 32;
	BS3d.y = 32;
	BS3d.z = 1;

	// init 3d thread grids	on GPU
	GS3d0.x = ceil(probe_shape * probe_shape / (float)BS3d.x);
	GS3d0.y = ceil(nscan / (float)BS3d.y);
	GS3d0.z = ceil(ntheta / (float)BS3d.z);

	GS3d1.x = ceil(detector_shape * detector_shape / (float)BS3d.x);
	GS3d1.y = ceil(nscan / (float)BS3d.y);
	GS3d1.z = ceil(ntheta / (float)BS3d.z);

	GS3d2.x = ceil(nscan / (float)BS3d.x);
	GS3d2.y = ceil(ntheta / (float)BS3d.y);
	GS3d2.z = 1;
}

// destructor, memory deallocation
ptychofft::~ptychofft()
{
  free();
}

void ptychofft::free()
{
  if(!is_free)
  {
    cufftDestroy(plan2d);
    cudaFree(fft_out);
    is_free = true;
  }
}

// forward ptychography operator g = FQf
void ptychofft::fwd(size_t g_, size_t f_, size_t scan_, size_t prb_)
{
  // convert pointers to correct type
  f = (float2 *)f_;
  g = (float2 *)g_;
  scan = (float2 *)scan_;
  prb = (float2 *)prb_;

	// probe multiplication of the object array
  cudaMemset(fft_out, 0, ntheta * nscan * detector_shape * detector_shape * sizeof(float2));
	muloperator<<<GS3d0, BS3d>>>(f, fft_out, prb, scan, ntheta, nz, n, nscan, probe_shape, detector_shape, 2); //flg==2 forward transform
	// Fourier transform
	cufftExecC2C(plan2d, (cufftComplex *)fft_out, (cufftComplex *)g, CUFFT_FORWARD);
}

// adjoint ptychography operator with respect to object (flg==0) f = Q*F*g, or probe (flg==1) prb = Q*F*g
void ptychofft::adj(size_t f_, size_t g_, size_t scan_, size_t prb_, int flg)
{
  // convert pointers to correct type
  f = (float2 *)f_;
  g = (float2 *)g_;
  scan = (float2 *)scan_;
  prb = (float2 *)prb_;

	// inverse Fourier transform
	cufftExecC2C(plan2d, (cufftComplex *)g, (cufftComplex *)fft_out, CUFFT_INVERSE);
	// adjoint probe (flg==0) or object (flg=1) multiplication operator
	muloperator<<<GS3d0, BS3d>>>(f, fft_out, prb, scan, ntheta, nz, n, nscan, probe_shape, detector_shape, flg);
}

Propagation::Propagation(size_t nwaves, size_t detector_shape,
                         size_t probe_shape)
    : nwaves(nwaves), detector_shape(detector_shape), probe_shape(probe_shape) {
  // Set number of waves to be processed in a GPU batch
  bwaves = min(nwaves, (size_t)512);
  // create batched 2D FFT plan on GPU with sizes (detector_shape,
  // detector_shape)
  // transform shape MUST be less than or equal to input and ouput shapes.
  int ffts[2] = {(int)detector_shape, (int)detector_shape };
  cufftPlanMany(&plan2d, 2, ffts,                         // transform shape
                ffts, 1, detector_shape * detector_shape, // input shape
                ffts, 1, detector_shape * detector_shape, // output shape
                CUFFT_C2C, bwaves // Number of FFTs to do simultaneously
                );
  // compute the FFT normalization constant.
  fft_norm = 1.0f / static_cast<float>(detector_shape);

  // Set CUDA kernel block size
  BS3d.x = 32;
  BS3d.y = 32;
  BS3d.z = 1;

  // Set CUDA kernel grid size
  GS3d0.x = ceil(detector_shape / (float)BS3d.x);
  GS3d0.y = ceil(detector_shape / (float)BS3d.y);
  GS3d0.z = bwaves;

  f_size = probe_shape * probe_shape * sizeof(float2);
  g_size = detector_shape * detector_shape * sizeof(float2);
  if ( cudaSuccess != cudaMalloc((void **)&_f, bwaves * f_size))
    printf("Error! Batch size too large for Propagation.\n");
  if ( cudaSuccess != cudaMalloc((void **)&_g, bwaves * g_size))
    printf("Error! Batch size too large for Propagation.\n");
}

Propagation::~Propagation() {
  free();
}

void Propagation::free()
{
  if(!is_free)
  {
    cufftDestroy(plan2d);
    cudaFree(_f);
    cudaFree(_g);
    is_free = true;
  }
}

void Propagation::fwd(size_t nearplane, size_t farplane) {
  for (size_t batch = 0; batch < nwaves; batch += bwaves) {
    size_t n_index = batch * probe_shape * probe_shape;
    size_t f_index = batch * detector_shape * detector_shape;
    _fwd((float2 *)nearplane + n_index, (float2 *)farplane + f_index,
         min(bwaves, nwaves - batch));
  }
}

void Propagation::_fwd(const float2 *f, float2 *g, size_t b) {
  cudaMemcpy(_f, f, b * f_size, cudaMemcpyHostToDevice);
  fft_pad <<<GS3d0, BS3d>>>
      (_f, _g, fft_norm, true, b, probe_shape, detector_shape);
  cufftExecC2C(plan2d, _g, _g, CUFFT_FORWARD);
  cudaMemcpy(g, _g, b * g_size, cudaMemcpyDeviceToHost);
}

void Propagation::adj(size_t nearplane, size_t farplane) {
  for (size_t batch = 0; batch < nwaves; batch += bwaves) {
    size_t n_index = batch * probe_shape * probe_shape;
    size_t f_index = batch * detector_shape * detector_shape;
    _adj((float2 *)nearplane + n_index, (float2 *)farplane + f_index,
         min(bwaves, nwaves - batch));
  }
}

void Propagation::_adj(float2 *f, const float2 *g, size_t b) {
  cudaMemcpy(_g, g, b * g_size, cudaMemcpyHostToDevice);
  cufftExecC2C(plan2d, (cufftComplex *)_g, (cufftComplex *)_g, CUFFT_INVERSE);
  fft_pad <<<GS3d0, BS3d>>>
      (_f, _g, fft_norm, false, b, probe_shape, detector_shape);
  cudaMemcpy(f, _f, b * f_size, cudaMemcpyDeviceToHost);
}

Convolution::Convolution(size_t probe_shape, size_t nscan, size_t nz, size_t n,
                         size_t ntheta)
    : ntheta(ntheta), nz(nz), n(n), nscan(nscan), probe_shape(probe_shape) {
  // Set the GPU batch sizes
  bscan = min(nscan, (size_t)256);

  // Set CUDA kernel block size
  BS3d.x = 512; // 512 is the maximum size limited by hardware.
  BS3d.y = 1;
  BS3d.z = 1;

  // Set CUDA kernel grid size
  GS3d0.x = ceil((probe_shape * probe_shape) / (float)BS3d.x);
  GS3d0.y = bscan;
  GS3d0.z = 1;

  scan_size = sizeof(float2);
  obj_size = nz * n * sizeof(float2);
  nearplane_size = probe_shape * probe_shape * sizeof(float2);
  // FIXME: Kernels will just fail to do anything if these allocations fail. 
  if ( cudaSuccess != cudaMalloc((void **)&_scan, bscan * scan_size))
    printf("Error! Batch size too large for Convolution.\n");
  if ( cudaSuccess != cudaMalloc((void **)&_obj, bscan * obj_size))
    printf("Error! Batch size too large for Convolution.\n");
  if ( cudaSuccess != cudaMalloc((void **)&_nearplane, bscan * nearplane_size))
    printf("Error! Batch size too large for Convolution.\n");
}

Convolution::~Convolution() {
  free();
}

void Convolution::free()
{
  if(!is_free)
  {
    cudaFree(_scan);
    cudaFree(_obj);
    cudaFree(_nearplane);
    is_free = true;
  }
}

void Convolution::fwd(size_t nearplane, size_t obj, size_t scan) {
  for (int view = 0; view < ntheta; view++) {
    for (int batch = 0; batch < nscan; batch += bscan) {
      size_t n_index = probe_shape * probe_shape * (batch + view * nscan);
      size_t o_index = view * nz * n;
      size_t s_index = batch + view * nscan;
      _fwd((float2 *)nearplane + n_index, (float2 *)obj + o_index,
           (float2 *)scan + s_index, min(bscan, nscan - batch));
    }
  }
}

void Convolution::_fwd(float2 *nearplane, const float2 *obj, const float2 *scan,
                       size_t b) {
  cudaMemcpy(_obj, obj, obj_size, cudaMemcpyHostToDevice);
  cudaMemcpy(_scan, scan, b * scan_size, cudaMemcpyHostToDevice);
  cudaMemcpy(_nearplane, nearplane, b * nearplane_size, cudaMemcpyHostToDevice);
  patch <<<GS3d0, BS3d>>>
      (_obj, _nearplane, _scan, 1, nz, n, probe_shape, b, true);
  cudaMemcpy(nearplane, _nearplane, b * nearplane_size, cudaMemcpyDeviceToHost);
}

void Convolution::adj(size_t nearplane, size_t obj, size_t scan) {
  for (int view = 0; view < ntheta; view++) {
    for (int batch = 0; batch < nscan; batch += bscan) {
      size_t n_index = probe_shape * probe_shape * (batch + view * nscan);
      size_t o_index = view * nz * n;
      size_t s_index = batch + view * nscan;
      _adj((float2 *)nearplane + n_index, (float2 *)obj + o_index,
           (float2 *)scan + s_index, min(bscan, nscan - batch));
    }
  }
}

void Convolution::_adj(const float2 *nearplane, float2 *obj, const float2 *scan,
                       size_t b) {
  cudaMemcpy(_obj, obj, obj_size, cudaMemcpyHostToDevice);
  cudaMemcpy(_scan, scan, b * scan_size, cudaMemcpyHostToDevice);
  cudaMemcpy(_nearplane, nearplane, b * nearplane_size, cudaMemcpyHostToDevice);
  patch <<<GS3d0, BS3d>>>
      (_obj, _nearplane, _scan, 1, nz, n, probe_shape, b, false);
  cudaMemcpy(obj, _obj, obj_size, cudaMemcpyDeviceToHost);
}
