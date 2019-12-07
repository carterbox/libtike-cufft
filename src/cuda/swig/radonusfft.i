/*interface*/
%module radonusfft

%{
#define SWIG_FILE_WITH_INIT
#include "radonusfft.cuh"
%}

class radonusfft
{
public:
  %immutable;
  size_t n;
  size_t ntheta;
  size_t nz;
  float center;

  %mutable;
  radonusfft(size_t ntheta, size_t nz, size_t n, float center);
  ~radonusfft();
  void fwd(size_t g, size_t f, size_t theta);
  void adj(size_t f, size_t g, size_t theta);
  void free();
};
