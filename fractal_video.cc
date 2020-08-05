#include "common.h"
#include <atomic>
#include <thread>

int main(int argc, char **argv) {
  if (argc < 2) {
    return 1;
  }
  auto weights = InitWeights(argc < 3 ? 0 : atoi(argv[2]));

  int num_threads = std::thread::hardware_concurrency();
  std::vector<Point> points(num_threads);

  auto init = std::chrono::high_resolution_clock::now();
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> start(-1, 1);
  for (size_t i = 0; i < num_threads; i++) {
    points[i].x = start(rng);
    points[i].y = start(rng);
    points[i].z = start(rng);
  }
  size_t iters = kTotalIters / num_threads;
  std::vector<std::atomic<Pixel>> pixels(kWidth * kHeight * kNumFrames);
  std::atomic<uint32_t> max{0};
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < num_threads; i++) {
    std::mt19937 rng(i);
    std::uniform_int_distribution<int> dist(0, kNumFunctions - 1);
    size_t local_max = 0;
    for (size_t j = 0; j < iters; j++) {
      size_t f = dist(rng);
      points[i] = Function()(points[i], weights[f].data());
      int x = std::round((points[i].x + 1.0f) * 0.5f * kWidth);
      int y = std::round((points[i].y + 1.0f) * 0.5f * kHeight);
      int z = std::round((points[i].z + 1.0f) * 0.5f * kNumFrames);
      if (x < 0 || y < 0 || z < 0 || x >= kWidth || y >= kHeight ||
          z >= kNumFrames) {
        continue;
      }
      size_t pos = (z * kHeight + y) * kWidth + x;
      Pixel p = pixels[pos];
      Pixel n;
      do {
        n = p.Add(kColors[f]);
      } while (!pixels[pos].compare_exchange_strong(p, n));
      if (((Pixel)pixels[pos]).count > local_max) {
        local_max = ((Pixel)pixels[pos]).count;
      }
    }
    uint32_t current_max = max;
    do {
      if (local_max < current_max) {
        break;
      }
    } while (!max.compare_exchange_strong(current_max, local_max));
  }
  auto count = std::chrono::high_resolution_clock::now();
  fprintf(stderr, "Highest count: %u\n", (uint32_t)max);
  float inv_log_max = 1.0f / std::log((uint32_t)max);
  float gamma = 1.0f / 4.0f;
  std::vector<uint8_t> images(4 * kWidth * kHeight * kNumFrames);
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < kNumFrames * kWidth * kHeight; i++) {
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
    images[4 * i] = std::round(std::max(0.0f, std::min(255.0f, r)));
    images[4 * i + 1] = std::round(std::max(0.0f, std::min(255.0f, g)));
    images[4 * i + 2] = std::round(std::max(0.0f, std::min(255.0f, b)));
  }
  auto render = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < kNumFrames; i++) {
    char buffer[1000];
    sprintf(buffer, "%s%05lu.png", argv[1], i);
    WritePNG(images.data() + 4 * kWidth * kHeight * i, kWidth, kHeight, buffer);
  }
  auto png = std::chrono::high_resolution_clock::now();

  fprintf(stderr, "Count: %8.3fms\n",
          std::chrono::duration_cast<std::chrono::microseconds>(count - init)
                  .count() *
              0.001f);

  fprintf(stderr, "Render: %8.3fms\n",
          std::chrono::duration_cast<std::chrono::microseconds>(render - count)
                  .count() *
              0.001f);

  fprintf(stderr, "PNG: %8.3fms\n",
          std::chrono::duration_cast<std::chrono::microseconds>(png - render)
                  .count() *
              0.001f);
}
