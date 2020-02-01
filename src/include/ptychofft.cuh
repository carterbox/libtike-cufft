#include <cufft.h>

#ifndef _PTYCHOFFT
#define _PTYCHOFFT

class ptychofft
{
  bool is_free = false;

	float2 *f;		// object
	float2 *g;		// data
	float2 *prb;	// probe function
	float2 *scan;   // x,y scan positions
  // Negative scan positions are skipped in kernel executions.

	cufftHandle plan2d;		 // 2D FFT plan
  float2 *fft_out;       // Buffer to store FFT output

	dim3 BS3d; // 3d thread block on GPU

	// 3d thread grids on GPU for different kernels
	dim3 GS3d0;
	dim3 GS3d1;
	dim3 GS3d2;

public:
  size_t ntheta; // number of projections
  size_t nz;	 // object vertical size
  size_t n;	  // object horizontal size
  size_t nscan;  // number of scan positions for 1 projection
  size_t detector_shape;  // detector size in 1 dimension
  size_t probe_shape;   // probe size in 1 dimension

	// constructor, memory allocation
	ptychofft(size_t ntheta, size_t nz, size_t n,
			  size_t nscan, size_t detector_shape, size_t probe_shape);
	// destructor, memory deallocation
	~ptychofft();
	// forward ptychography operator FQ
	void fwd(size_t g_, size_t f_, size_t scan_, size_t prb_);
	// adjoint ptychography operator with respect to object (fgl==0) f = Q*F*g, or probe (flg==1) prb = Q*F*g
	void adj(size_t f_, size_t g_, size_t scan_, size_t prb_, int flg);
  void free();
};

class Propagation {
  size_t bwaves;      // Number of waves to process simultaneously on GPU
  cufftHandle plan2d; // 2D FFT plan
  float fft_norm;     // FFT normalization constant
  dim3 BS3d;          // 3d thread block on GPU
  dim3 GS3d0;         // 3d thread grids on GPU

  size_t f_size; // size of single nearplane wave
  size_t g_size; // size of single farpane wave
  float2 *_f;    // pointer to GPU memory nearplane
  float2 *_g;    // pointer to GPU memory farplane

  void _fwd(const float2 *, float2 *, size_t);
  void _adj(float2 *, const float2 *, size_t);

public:
  size_t nwaves;         // number waves to propagate
  size_t detector_shape; // detector size in 1 dimension
  size_t probe_shape;    // probe size in 1 dimension

  Propagation(size_t nwaves, size_t detector_shape, size_t probe_shape);
  ~Propagation();

  void fwd(size_t nearplane, size_t farplane);
  void adj(size_t nearplane, size_t farplane);
};

class Convolution {
  dim3 BS3d;    // 3d thread block on GPU
  dim3 GS3d0;   // 3d thread grids on GPU
  size_t bscan; // number of scan positions to process simultaneously on GPU

  size_t scan_size;      // size of single scan position
  size_t obj_size;       // size of single object plane
  size_t nearplane_size; // size of single nearplane wave
  float2 *_scan;         // pointer to GPU memory scan positions
  float2 *_obj;          // pointer to GPU memory object plane
  float2 *_nearplane;    // pointer to GPU memory nearplane waves

  void _fwd(float2 *, const float2 *, const float2 *, size_t);
  void _adj(const float2 *, float2 *, const float2 *, size_t);

public:
  size_t ntheta;      // number of projections
  size_t nz;          // object vertical size
  size_t n;           // object horizontal size
  size_t nscan;       // number of scan positions for 1 projection
  size_t probe_shape; // probe size in 1 dimension

  Convolution(size_t probe_shape, size_t nscan, size_t nz, size_t n,
              size_t ntheta);
  ~Convolution();

  void fwd(size_t nearplane, size_t obj, size_t scan);
  void adj(size_t nearplane, size_t obj, size_t scan);
};

#endif
