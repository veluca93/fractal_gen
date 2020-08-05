#include "common.h"
#include <atomic>
#include <cuda_runtime.h>
#include <string.h>
#include <thread>

__global__ void TransformToImage(uint32_t *max, float gamma,
                                 const Pixel *pixels, uint8_t *images) {
  float inv_log_max = 1.0f / logf(*max);
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < kNumFrames * kWidth * kHeight; i += stride) {
    int z = i / (kWidth * kHeight);
    int y = i % (kWidth * kHeight) / kWidth;
    int x = i % kWidth;
    float r = 0, g = 0, b = 0;

    size_t count = ((Pixel)pixels[i]).count;
    float base_frac = count == 0 ? 0 : logf(count) * inv_log_max;
    float denom = 0;
    for (int iz = -kKernelRadius; iz <= kKernelRadius; iz++) {
      if ((iz + z) < 0 || (iz + z) >= kNumFrames) {
        continue;
      }
      for (int iy = -kKernelRadius; iy <= kKernelRadius; iy++) {
        if ((iy + y) < 0 || (iy + y) >= kHeight) {
          continue;
        }
        for (int ix = -kKernelRadius; ix <= kKernelRadius; ix++) {
          if ((ix + x) < 0 || (ix + x) >= kWidth) {
            continue;
          }
          size_t pos = (z * kHeight + y + iy) * kWidth + x + ix;
          const Pixel &p = pixels[pos];
          float frac = p.count == 0 ? 0 : logf(p.count) * inv_log_max;
          float scale =
              expf(-kDistMultiplier * sqrtf(ix * ix + iy * iy + iz * iz) *
                   (base_frac) / (frac + 1e-5));
          float mul = p.count == 0 ? 0 : powf(frac, gamma);
          r += p.color.r * mul * scale;
          g += p.color.g * mul * scale;
          b += p.color.b * mul * scale;
          denom += scale;
        }
      }
    }
    denom = 1.0f / denom;
    r *= denom;
    g *= denom;
    b *= denom;
    images[4 * i] = r < 0.0f ? 0.0f : r > 255.0f ? 255.0f : r;
    images[4 * i + 1] = g < 0.0f ? 0.0f : g > 255.0f ? 255.0f : g;
    images[4 * i + 2] = b < 0.0f ? 0.0f : b > 255.0f ? 255.0f : b;
    images[4 * i + 3] = 255;
  }
}

__device__ void atomicAddPixel(Pixel *address, Color val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    Pixel assumed_pixel = *reinterpret_cast<Pixel *>(&assumed);
    Pixel new_pixel = assumed_pixel.Add(val);
    unsigned long long int nw =
        *reinterpret_cast<unsigned long long int *>(&new_pixel);
    old = atomicCAS(address_as_ull, assumed, nw);
  } while (assumed != old);
}

__global__ void ComputePixels(Pixel *pixels, size_t iters, uint32_t *max,
                              const Color *colors, const float *weights) {
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t s[2] = {1U, index};
  auto result = [&]() {
    uint64_t s1 = s[0];
    const uint64_t s0 = s[1];
    const uint64_t result = s0 + s1;
    s[0] = s0;
    s1 ^= s1 << 23;                          // a
    s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5); // b, c
    return result;
  };

  constexpr size_t rng_range = 1024 * 1024;
  Point p;
  p.x = (result() % rng_range) * 2.0f - 1.0f;
  p.y = (result() % rng_range) * 2.0f - 1.0f;
  p.z = (result() % rng_range) * 2.0f - 1.0f;

  size_t local_max = 0;
  for (size_t j = 0; j < iters; j++) {
    size_t f = result() % kNumFunctions;
    p = Function()(p, weights + 13 * Function::kNumBases * f);
    int x = std::round((p.x + 1.0f) * 0.5f * kWidth);
    int y = std::round((p.y + 1.0f) * 0.5f * kHeight);
    int z = std::round((p.z + 1.0f) * 0.5f * kNumFrames);
    if (x < 0 || y < 0 || z < 0 || x >= kWidth || y >= kHeight ||
        z >= kNumFrames) {
      continue;
    }
    size_t pos = (z * kHeight + y) * kWidth + x;
    atomicAddPixel(&pixels[pos], colors[f]);
    if (((Pixel)pixels[pos]).count > local_max) {
      local_max = ((Pixel)pixels[pos]).count;
    }
  }
  atomicMax(max, local_max);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    return 1;
  }
  auto weights = InitWeights(argc < 3 ? 0 : atoi(argv[2]));

  static_assert(sizeof(Pixel) == sizeof(unsigned long long),
                "Size of pixel is wrong");

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  size_t tpb = 1024;
  size_t blks = prop.multiProcessorCount;
  size_t num_threads = tpb * blks;
  Pixel *pixels_cuda;
  cudaMallocManaged(&pixels_cuda,
                    kWidth * kHeight * kNumFrames * sizeof(Pixel));
  memset(pixels_cuda, 0, kWidth * kHeight * kNumFrames * sizeof(Pixel));
  size_t iters = (kTotalIters + num_threads - 1) / num_threads;
  uint32_t *max_cuda;
  cudaMallocManaged(&max_cuda, sizeof(uint32_t));
  Color *colors;
  cudaMallocManaged(&colors, sizeof(kColors));
  memcpy(colors, kColors.data(), sizeof(kColors));
  float *weights_cuda;
  cudaMallocManaged(&weights_cuda, sizeof(weights));
  memcpy(weights_cuda, weights.data(), sizeof(weights));
  *max_cuda = 0;
  ComputePixels<<<tpb, blks>>>(pixels_cuda, iters, max_cuda, colors,
                               weights_cuda);
  float gamma = 1.0f / 4.0f;

  uint8_t *images;
  cudaMallocManaged(&images, 4 * kWidth * kHeight * kNumFrames);

  int blockSize = 256;
  int numBlocks = (kWidth * kHeight * kNumFrames + blockSize - 1) / blockSize;

  TransformToImage<<<numBlocks, blockSize>>>(max_cuda, gamma, pixels_cuda,
                                             images);
  cudaDeviceSynchronize();
  fprintf(stderr, "Highest count: %u\n", *max_cuda);

  auto write_frame = [&](size_t i) {
    char buffer[1000];
    sprintf(buffer, "%s%05lu.png", argv[1], i);
    WritePNG(images + 4 * kWidth * kHeight * i, kWidth, kHeight, buffer);
  };
  std::vector<std::thread> threads;
  for (size_t i = 0; i < kNumFrames; i++) {
    threads.emplace_back(write_frame, i);
  }
  for (size_t i = 0; i < kNumFrames; i++) {
    threads[i].join();
  }
  cudaFree(pixels_cuda);
  cudaFree(images);
}
