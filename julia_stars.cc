#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <random>
#include <stdio.h>
#include <vector>

#include "common.h"

constexpr size_t kSize = 4096;
constexpr size_t kMaxIters = 64;

int main() {
  std::vector<unsigned char> img(kSize * kSize * 3);
  std::vector<unsigned char> star_size(kSize * kSize);
  std::vector<float> star_alpha(kSize * kSize);
  std::complex<float> c{0.11, 0.550};
  std::mt19937 rng;
  std::uniform_real_distribution<float> dist(0, 1);
#pragma omp parallel for
  for (size_t y = 0; y < kSize; y++) {
    for (size_t x = 0; x < kSize; x++) {
      float range = 1.5f;
      std::complex<float> start{2 * range * x / kSize - range,
                                2 * range * y / kSize - range};
      size_t iters = 0;
      float col = 0;
      for (; iters < kMaxIters && norm(start) < 30; iters++) {
        start = start * start + c;
        col += std::exp(-std::norm(start));
      }
      float prob = 0.5f * (1.0f + std::sin(col));
      if (iters == kMaxIters && prob > 0.5f) {
        prob = 1.0f - prob;
      }
      float r, g, b;
      b = prob * prob;
      r = b * b;
      g = r * b;
      if (iters == kMaxIters && dist(rng) < std::pow(prob, 1.0) / 30) {
        star_size[y * kSize + x] =
            dist(rng) < 0.5 ? (dist(rng) < 0.5 ? 3 : 2) : 1;
        if (star_size[y * kSize + x]) {
          star_alpha[y * kSize + x] = dist(rng) * 0.1 + 0.9;
        } else {
          star_alpha[y * kSize + x] = 1;
        }
      }
      img[y * kSize * 3 + x * 3 + 0] =
          std::min(255.0f, std::max(0.0f, std::round(r * 255)));
      img[y * kSize * 3 + x * 3 + 1] =
          std::min(255.0f, std::max(0.0f, std::round(g * 255)));
      img[y * kSize * 3 + x * 3 + 2] =
          std::min(255.0f, std::max(0.0f, std::round(b * 255)));
    }
  }
#pragma omp parallel for
  for (int y0 = 0; y0 < kSize; y0++) {
    for (int x0 = 0; x0 < kSize; x0++) {
      size_t size = star_size[y0 * kSize + x0];
      if (size == 0)
        continue;
      size--;
      float alpha = star_alpha[y0 * kSize + x0];
      int lowy = y0 >= size ? y0 - size : 0;
      int lowx = x0 >= size ? x0 - size : 0;
      int hiy = y0 + size < kSize ? y0 + size : kSize - 1;
      int hix = x0 + size < kSize ? x0 + size : kSize - 1;
      for (int y = lowy; y <= hiy; y++) {
        for (int x = lowx; x <= hix; x++) {
          if (std::abs(x - x0) + std::abs(y - y0) > size)
            continue;
          img[y * kSize * 3 + x * 3 + 0] =
              (1 - alpha) * img[y * kSize * 3 + x * 3 + 0] + alpha * 255;
          img[y * kSize * 3 + x * 3 + 1] =
              (1 - alpha) * img[y * kSize * 3 + x * 3 + 1] + alpha * 255;
          img[y * kSize * 3 + x * 3 + 2] =
              (1 - alpha) * img[y * kSize * 3 + x * 3 + 2] + alpha * 255;
        }
      }
    }
  }

  WritePNG(img.data(), kSize, kSize, "julia_stars.png");
}
