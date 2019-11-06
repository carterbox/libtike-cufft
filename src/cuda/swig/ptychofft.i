/*interface*/
%module ptychofft

%{
#define SWIG_FILE_WITH_INIT
#include "ptychofft.cuh"
%}

class ptychofft {
private:
  bool is_free = false;
  float2 *f;   // complex object. (ntheta, nz, n)
  float2 *g;   // complex data. (ntheta, nscan, detector_shape, detector_shape)
  float2 *prb; // complex probe function. (nethat, nsca, probe_shape, probe_shape)
  float *scan; // vertical, horizonal scan positions. (ntheta, nscan, 2)
               // Negative scan positions are skipped in kernel executions.
  cufftHandle plan2d; // 2D FFT plan
  dim3 BS3d; // 3d thread block on GPU
  dim3 GS3d0, GS3d1, GS3d2;
  size_t ndetx, ndety, nprb;
public:
  %immutable;
  size_t ntheta;        // number of projections
  size_t nz;            // object vertical size
  size_t n;             // object horizontal size
  size_t nscan;         // number of scan positions for 1 projection
  size_t detector_shape; // detector width and height
  size_t probe_shape;    // probe size in 1 dimension
  %mutable;
  ptychofft(size_t ntheta, size_t nz, size_t n, size_t nscan,
            size_t detector_shape, size_t probe_shape);
  ~ptychofft();
  void free();
  void fwd(size_t g, size_t f, size_t scan, size_t prb);
  void adj(size_t g, size_t f, size_t scan, size_t prb, int flg);
};
