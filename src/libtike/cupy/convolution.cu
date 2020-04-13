// Extract patches from an image at scan locations
// OR add patches to an image at scan locations
//
// This function extracts patches using linear interpolation at each of the scan
// points.
// Assuming square patches, but rectangular image.
// nscan is the number of positions per images
// scan has shape (nimage, nscan)
// images has shape (nimage, nimagey, nimagex)
extern "C" __global__
void patch(float2 *images, float2 *patches, const float2 *scan,
           int nimage, int nimagey, int nimagex, int patch_shape,
           int nscan, bool forward) {
  const int tx = threadIdx.x + blockDim.x * (blockIdx.x);
  const int ty = blockIdx.y;
  const int tz = blockIdx.z;
  if (tx >= patch_shape * patch_shape || ty >= nscan || tz >= nimage) return;

  float sx; // modf requires a place to save the integer part
  float sy;
  const float sxf = modff(scan[ty + tz * nscan].y, &sx);
  const float syf = modff(scan[ty + tz * nscan].x, &sy);

  // skip scans where the probe position overlaps edges
  if (sx < 0 || nimagex <= sx + patch_shape || sy < 0 ||
      nimagey <= sy + patch_shape)
    return;

  // image index (ii)
  const int ii = (
    sx + (tx % patch_shape)
    + nimagex * ((sy + tx / patch_shape) + nimagey * tz)
  );
  // patch index (pi)
  const int pi = tx + patch_shape * patch_shape * (ty + nscan * (tz));

  // Linear interpolation
  if (forward) {
    patches[pi].x = images[ii              ].x * (1 - sxf) * (1 - syf)
                  + images[ii + 1          ].x * (    sxf) * (1 - syf)
                  + images[ii     + nimagex].x * (1 - sxf) * (    syf)
                  + images[ii + 1 + nimagex].x * (    sxf) * (    syf);

    patches[pi].y = images[ii              ].y * (1 - sxf) * (1 - syf)
                  + images[ii + 1          ].y * (    sxf) * (1 - syf)
                  + images[ii     + nimagex].y * (1 - sxf) * (    syf)
                  + images[ii + 1 + nimagex].y * (    sxf) * (    syf);
  } else {
    const float2 tmp = patches[pi];
    atomicAdd(&images[ii              ].x, tmp.x * (1 - sxf) * (1 - syf));
    atomicAdd(&images[ii              ].y, tmp.y * (1 - sxf) * (1 - syf));
    atomicAdd(&images[ii + 1          ].y, tmp.y * (    sxf) * (1 - syf));
    atomicAdd(&images[ii + 1          ].x, tmp.x * (    sxf) * (1 - syf));
    atomicAdd(&images[ii     + nimagex].x, tmp.x * (1 - sxf) * (    syf));
    atomicAdd(&images[ii     + nimagex].y, tmp.y * (1 - sxf) * (    syf));
    atomicAdd(&images[ii + 1 + nimagex].x, tmp.x * (    sxf) * (    syf));
    atomicAdd(&images[ii + 1 + nimagex].y, tmp.y * (    sxf) * (    syf));
  }
}