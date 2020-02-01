#include "ptychofft.cuh"
#include "kernels.cu"

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
  // create batched 2D FFT plan on GPU with sizes (detector_shape,
  // detector_shape)
  // transform shape MUST be less than or equal to input and ouput shapes.
  int ffts[2] = {(int)detector_shape, (int)detector_shape };
  cufftPlanMany(&plan2d, 2, ffts,                         // transform shape
                ffts, 1, detector_shape * detector_shape, // input shape
                ffts, 1, detector_shape * detector_shape, // output shape
                CUFFT_C2C, nwaves // Number of FFTs to do simultaneously
                );
  // create a place to put the FFT and IFFT output.
  cudaMalloc((void **)&fft_buffer,
             nwaves * detector_shape * detector_shape * sizeof(float2));
  // compute the FFT normalization constant.
  fft_norm = 1.0f / static_cast<float>(detector_shape);

  // Set CUDA kernel block size
  BS3d.x = 32;
  BS3d.y = 32;
  BS3d.z = 1;

  // Set CUDA kernel grid size
  GS3d0.x = ceil(detector_shape / (float)BS3d.x);
  GS3d0.y = ceil(detector_shape / (float)BS3d.y);
  GS3d0.z = nwaves;
}

Propagation::~Propagation(){
  cufftDestroy(plan2d);
  cudaFree(fft_buffer);
}

void Propagation::fwd(size_t f, size_t g) {
  fft_pad<<<GS3d0, BS3d>>>((float2 *)f, fft_buffer, fft_norm,
                           true, nwaves, probe_shape, detector_shape);
  cufftExecC2C(plan2d, (cufftComplex *)fft_buffer, (cufftComplex *)g, CUFFT_FORWARD);
}

void Propagation::adj(size_t f, size_t g) {
  cufftExecC2C(plan2d, (cufftComplex *)g, (cufftComplex *)fft_buffer, CUFFT_INVERSE);
  fft_pad<<<GS3d0, BS3d>>>((float2 *)f, fft_buffer, fft_norm,
                           false, nwaves, probe_shape, detector_shape);
}
