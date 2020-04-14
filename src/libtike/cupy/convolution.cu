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

  // patch index (pi)
  const int pi = tx + patch_shape * patch_shape * (ty + nscan * (tz));

  const float sx = floor(scan[ty + tz * nscan].y);
  const float sy = floor(scan[ty + tz * nscan].x);

  if (sx < 0 || nimagex <= sx + patch_shape ||
      sy < 0 || nimagey <= sy + patch_shape){
    // printf("%f, %f - %f, %f\n", sx, sy, sxf, syf);
    // scans where the probe position overlaps edges we fill with zeros
    if (forward){
      patches[pi].x = 0.0f;
      patches[pi].y = 0.0f;
    }
    return;
  }

  // image index (ii)
  const int ii = (
    sx + (tx % patch_shape)
    + nimagex * ((sy + tx / patch_shape) + nimagey * tz)
  );

  const float sxf = scan[ty + tz * nscan].y - sx;
  const float syf = scan[ty + tz * nscan].x - sy;
  assert(1.0f >= sxf && sxf >= 0.0f && 1.0f >= syf && syf >= 0.0f);

  // Linear interpolation
  if (forward) {
    patches[pi].x = images[ii              ].x * (1.0f - sxf) * (1.0f - syf)
                  + images[ii + 1          ].x * (       sxf) * (1.0f - syf)
                  + images[ii     + nimagex].x * (1.0f - sxf) * (       syf)
                  + images[ii + 1 + nimagex].x * (       sxf) * (       syf);

    patches[pi].y = images[ii              ].y * (1.0f - sxf) * (1.0f - syf)
                  + images[ii + 1          ].y * (       sxf) * (1.0f - syf)
                  + images[ii     + nimagex].y * (1.0f - sxf) * (       syf)
                  + images[ii + 1 + nimagex].y * (       sxf) * (       syf);
  } else {
    const float2 tmp = patches[pi];
    atomicAdd(&images[ii              ].x, tmp.x * (1.0f - sxf) * (1.0f - syf));
    atomicAdd(&images[ii              ].y, tmp.y * (1.0f - sxf) * (1.0f - syf));
    atomicAdd(&images[ii + 1          ].y, tmp.y * (       sxf) * (1.0f - syf));
    atomicAdd(&images[ii + 1          ].x, tmp.x * (       sxf) * (1.0f - syf));
    atomicAdd(&images[ii     + nimagex].x, tmp.x * (1.0f - sxf) * (       syf));
    atomicAdd(&images[ii     + nimagex].y, tmp.y * (1.0f - sxf) * (       syf));
    atomicAdd(&images[ii + 1 + nimagex].x, tmp.x * (       sxf) * (       syf));
    atomicAdd(&images[ii + 1 + nimagex].y, tmp.y * (       sxf) * (       syf));
  }
}