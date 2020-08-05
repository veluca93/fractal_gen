#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <stdio.h>
#include <vector>

#include "common.h"

static constexpr size_t kNumVariations = 4;
static constexpr size_t kSize = 2048 * 2;
static constexpr size_t kSupersample = 4;
static constexpr float kGamma = 4;
static constexpr size_t kNumIters = 1ULL << 32;

struct Point2d {
  float rgb[3];
  float x;
  float y;
};

struct Function2d {
  struct VariationParams {
    float w = 0;                                    // Weight
    float a = 0, b = 0, c = 0, d = 0, e = 0, f = 0; // Affine params
    void Read() { scanf("%f%f%f%f%f%f%f", &w, &a, &b, &c, &d, &e, &f); }
    std::pair<float, float> Affine(float x, float y) {
      return {a * x + b * y + c, d * x + e * y + f};
    }
  };
  float probability;
  float rgb[3] = {};
  VariationParams variations[kNumVariations] = {};
  void Read() {
    int num_variations;
    scanf("%f%f%f%f%d", &probability, &rgb[0], &rgb[1], &rgb[2],
          &num_variations);
    assert(num_variations <= kNumVariations);
    for (int i = 0; i < num_variations; i++) {
      variations[i].Read();
    }
  }
  void Update(Point2d *point) {
    auto v0 = variations[0].Affine(point->x, point->y);
    auto v1 = variations[1].Affine(point->x, point->y);
    auto v2 = variations[2].Affine(point->x, point->y);
    auto v3 = variations[3].Affine(point->x, point->y);
    for (size_t i = 0; i < 3; i++) {
      point->rgb[i] = (rgb[i] + point->rgb[i]) / 2;
    }
    float v2_scale =
        1.0f / (v2.first * v2.first + v2.second * v2.second + 1e-6);
    float v3_scale =
        1.0f / std::sqrt(v3.first * v3.first + v3.second * v3.second + 1e-6);
    point->x = v0.first * variations[0].w +
               std::sin(v1.first) * variations[1].w +
               v2.first * v2_scale * variations[2].w +
               (v3.first * v3.first - v3.second * v3.second) * v3_scale *
                   variations[3].w;
    point->y = v0.second * variations[0].w +
               std::sin(v1.second) * variations[1].w +
               v2.second * v2_scale * variations[2].w +
               (2 * v3.first * v3.second) * v3_scale * variations[3].w;
  }
};

struct PixelState {
  float frequency = 1;
  float rgb[3] = {};
  void Add(const float color[3]) {
    frequency += 1;
    for (size_t i = 0; i < 3; i++) {
      rgb[i] = (color[i] + rgb[i]) / 2;
    }
  }
};

int main() {
  int num_functions;
  scanf("%d", &num_functions);
  std::vector<Function2d> functions(num_functions);
  for (size_t i = 0; i < num_functions; i++) {
    functions[i].Read();
  }
  std::vector<float> cdf(num_functions + 1);
  for (size_t i = 0; i < num_functions; i++) {
    cdf[i + 1] = cdf[i] + functions[i].probability;
  }
  for (size_t i = 0; i < num_functions + 1; i++) {
    cdf[i] /= cdf.back();
  }

  std::vector<PixelState> pixels(kSize * kSize * kSupersample * kSupersample);

  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(0, 1);
  Point2d point;
  point.x = dist(rng);
  point.y = dist(rng);
  point.rgb[0] = 255;
  point.rgb[1] = 0;
  point.rgb[2] = dist(rng) * 255;
  for (size_t i = 0; i < kNumIters; i++) {
    float prob = dist(rng);
    int idx = std::lower_bound(cdf.begin(), cdf.end(), prob) - cdf.begin() - 1;
    functions[idx].Update(&point);
    int x = (point.x + 1) * 0.5f * kSize * kSupersample;
    int y = (point.y + 1) * 0.5f * kSize * kSupersample;
    if (x >= kSize * kSupersample)
      continue;
    if (y >= kSize * kSupersample)
      continue;
    if (x < 0)
      continue;
    if (y < 0)
      continue;
    pixels[y * kSize * kSupersample + x].Add(point.rgb);
  }

  std::vector<PixelState> scaled_pixels(kSize * kSize);
  // Convert pixel states to colors.
  float max_freq = 0;
  for (size_t y = 0; y < kSize; y++) {
    for (size_t x = 0; x < kSize; x++) {
      scaled_pixels[y * kSize + x].frequency = 0;
      for (size_t j = 0; j < kSupersample; j++) {
        for (size_t k = 0; k < kSupersample; k++) {
          scaled_pixels[y * kSize + x].frequency +=
              pixels[(y * kSupersample + j) * kSize * kSupersample +
                     x * kSupersample + k]
                  .frequency;
          for (size_t i = 0; i < 3; i++) {
            scaled_pixels[y * kSize + x].rgb[i] +=
                pixels[(y * kSupersample + j) * kSize * kSupersample +
                       x * kSupersample + k]
                    .rgb[i];
          }
        }
      }
      scaled_pixels[y * kSize + x].frequency /= kSupersample * kSupersample;
      for (size_t i = 0; i < 3; i++) {
        scaled_pixels[y * kSize + x].rgb[i] /= kSupersample * kSupersample;
      }
      max_freq = std::max(scaled_pixels[y * kSize + x].frequency, max_freq);
    }
  }
  std::vector<unsigned char> img(kSize * kSize * 3);
  for (size_t y = 0; y < kSize; y++) {
    for (size_t x = 0; x < kSize; x++) {
      float alpha =
          std::log(scaled_pixels[y * kSize + x].frequency) / std::log(max_freq);
      for (size_t i = 0; i < 3; i++) {
        img[3 * kSize * y + 3 * x + i] =
            std::max(0.f, std::min(255.f, scaled_pixels[y * kSize + x].rgb[i] *
                                              std::pow(alpha, 1 / kGamma)));
      }
    }
  }

  WritePNG(img.data(), kSize, kSize, "flame.png");
}
