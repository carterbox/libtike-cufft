#include "ptychofft.cuh"
#include "kernels.cu"

// constructor, memory allocation
ptychofft::ptychofft(
  size_t detector_shape, size_t probe_shape, size_t nscan, size_t nz, size_t n,
  size_t ntheta
) :
  ntheta(ntheta), nz(nz), n(n), nscan(nscan),
  ndetx(detector_shape), ndety(detector_shape),
  nprb(probe_shape), probe_shape(probe_shape)
{
	// create batched 2d FFT plan on GPU with sizes (ndetx,ndety)
	int fft_det[2] = {(int)ndetx, (int)ndety};
  int fft_prb[2] = {(int)nprb, (int)nprb};
	cufftPlanMany(&plan2d, 2,
    fft_det, // transform shape
    fft_prb, // input shape
    1, nprb * nprb,
    fft_det, // output shape
    1, ndetx * ndety,
    CUFFT_C2C, // type
    ntheta * nscan  // Number of FFTs to do simultaneously
  );

	// init 3d thread block on GPU
	BS3d.x = 32;
	BS3d.y = 32;
	BS3d.z = 1;

	// init 3d thread grids	on GPU
	GS3d0.x = ceil(nprb * nprb / (float)BS3d.x);
	GS3d0.y = ceil(nscan / (float)BS3d.y);
	GS3d0.z = ceil(ntheta / (float)BS3d.z);

	GS3d1.x = ceil(ndetx * ndety / (float)BS3d.x);
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
    is_free = true;
  }
}

// forward ptychography operator g = FQf
void ptychofft::fwd(size_t g_, size_t f_, size_t scan_, size_t prb_)
{
  // convert pointers to correct type
  f = (float2 *)f_;
  g = (float2 *)g_;
  float *scanx = &((float *)scan_)[0];
  float *scany = &((float *)scan_)[ntheta * nscan];
  prb = (float2 *)prb_;

	// probe multiplication of the object array
	muloperator<<<GS3d0, BS3d>>>(f, g, prb, scanx, scany, ntheta, nz, n, nscan, nprb, ndetx, ndety, 2); //flg==2 forward transform
	// Fourier transform
	cufftExecC2C(plan2d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_FORWARD);
}

// adjoint ptychography operator with respect to object (flg==0) f = Q*F*g, or probe (flg==1) prb = Q*F*g
void ptychofft::adj(size_t g_, size_t f_, size_t scan_, size_t prb_, int flg)
{
  // convert pointers to correct type
  f = (float2 *)f_;
  g = (float2 *)g_;
  float *scanx = &((float *)scan_)[0];
  float *scany = &((float *)scan_)[ntheta * nscan];
  prb = (float2 *)prb_;

	// inverse Fourier transform
	cufftExecC2C(plan2d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_INVERSE);
	// adjoint probe (flg==0) or object (flg=1) multiplication operator
	muloperator<<<GS3d0, BS3d>>>(f, g, prb, scanx, scany, ntheta, nz, n, nscan, nprb, ndetx, ndety, flg);
}
