#include "omp.h"
#include <algorithm>
#include <array>
#include <assert.h>
#include <complex>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <vector>

#include "common.h"

using PatternT = std::vector<std::complex<double>>;
std::complex<double> Coord(int x, int y, size_t xsz, size_t ysz, float scale) {
  int cx = xsz / 2;
  int cy = ysz / 2;
  double real = (x - cx) * 2.0f / xsz * scale;
  double imag = (y - cy) * 2.0f / ysz * scale;
  return std::complex<double>(real, imag);
}

void ComputePattern(PatternT *pattern_img,
                    const std::vector<std::string> &pattern, size_t xsz,
                    size_t ysz, size_t pad_x, size_t pad_y, size_t pxsz,
                    size_t pysz, float scale) {
  size_t scaling = ysz / pysz;
  for (size_t y = 0; y < ysz; y++) {
    for (size_t x = 0; x < xsz; x++) {
      size_t px = x / scaling;
      size_t py = y / scaling;
      if (px < pad_x || px - pad_x >= pattern[0].size() || py < pad_y ||
          py - pad_y >= pattern.size() ||
          pattern[py - pad_y][px - pad_x] != '#') {
      } else {
        pattern_img->push_back(Coord(x, y, xsz, ysz, scale));
      }
    }
  }
}

std::complex<double> RemapCoord(const PatternT &pattern_img, int x, int y,
                                size_t xsz, size_t ysz, double scale,
                                bool *is_inside) {
  std::complex<double> cur = Coord(x, y, xsz, ysz, scale);
  std::complex<double> accum = 0;
  double total_weight = 0;
  *is_inside = false;
  for (std::complex<double> ref : pattern_img) {
    double d = std::norm(cur - ref);
#if 1
    constexpr size_t kDoublings = 8;
    double w = 1 - d * (1.0 / (1 << kDoublings));
    for (size_t i = 0; i < kDoublings; i++) {
      w *= w;
    }
#else
    double w = std::exp(-d);
#endif
    if (w >= 1 - 1e-5) {
      *is_inside = true;
    }
    accum += w * (cur - ref);
    total_weight += w;
  }
  return accum / total_weight;
}

std::complex<double> polyval(const std::vector<double> &poly,
                             std::complex<double> x) {
  std::complex<double> out = poly.back();
  for (int i = poly.size(); i > 1; i--) {
    out = out * x + poly[i - 2];
  }
  return out;
}

void ComputeFractal(const PatternT &pattern_img,
                    std::vector<unsigned char> *output, size_t xsz, size_t ysz,
                    double scale, const std::vector<double> &poly,
                    const std::vector<std::array<int, 3>> &colours) {
  std::vector<double> color_scale(xsz * ysz);
  std::vector<std::complex<double>> attractor(xsz * ysz);
  double infty = scale * 100;
  constexpr size_t kMaxItersL = 11;
  constexpr size_t kMinItersL = 3;
  constexpr size_t kMaxIters = 1 << kMaxItersL;
  constexpr size_t kMinIters = 1 << kMinItersL;
  constexpr double kConvergenceNorm = 1e-10;
  constexpr double kCloseRootNorm = 1e-4;
  size_t N = poly.size() - 1;
  std::vector<double> deriv(N);
  for (size_t i = 0; i < N; i++) {
    deriv[i] = poly[i + 1] * (i + 1);
  }
#pragma omp parallel for
  for (size_t y = 0; y < ysz; y++) {
    for (size_t x = 0; x < xsz; x++) {
      bool is_inside = false;
      auto p = RemapCoord(pattern_img, x, y, xsz, ysz, scale, &is_inside);
      auto last_p = p + 1000.0;
      size_t iters = kMinIters;
      for (; iters < kMaxIters; iters++) {
        last_p = p;
        p -= polyval(poly, p) / polyval(deriv, p);
        if (std::norm(p - last_p) < kConvergenceNorm)
          break;
      }
      if (iters < kMinIters) {
        iters = kMinIters;
      }
      double iters_mul =
          1.0f - (std::log2f(iters) - kMinItersL) / (kMaxItersL - kMinItersL);
      if (!is_inside) {
        iters_mul *= iters_mul;
      }
      color_scale[xsz * y + x] = iters_mul;
      attractor[xsz * y + x] =
          iters < kMaxIters ? p : std::complex<double>(infty + 1, infty + 1);
    }
  }

  std::vector<std::complex<double>> roots;
  auto closest_root = [&roots, infty](std::complex<double> in) {
    double dist = infty;
    size_t closest = 0;
    assert(!roots.empty());
    for (size_t i = 0; i < roots.size(); i++) {
      double d = std::norm(in - roots[i]);
      if (d < dist) {
        dist = d;
        closest = i;
      }
    }
    return closest;
  };
  for (size_t i = 0; i < attractor.size(); i++) {
    if (std::norm(attractor[i]) >= infty) {
      continue;
    }
    if (roots.empty()) {
      roots.push_back(attractor[i]);
      continue;
    }
    size_t close = closest_root(attractor[i]);
    if (std::norm(roots[close] - attractor[i]) >= kCloseRootNorm) {
      roots.push_back(attractor[i]);
    }
  }
  std::sort(roots.begin(), roots.end(), [](auto x, auto y) {
    return std::make_pair(std::arg(x), std::norm(x)) <
           std::make_pair(std::arg(y), std::norm(y));
  });
  for (size_t i = 0; i < roots.size(); i++) {
    std::cerr << roots[i] << std::endl;
  }
  assert(roots.size() <= colours.size());
#pragma omp parallel for
  for (size_t y = 0; y < ysz; y++) {
    for (size_t x = 0; x < xsz; x++) {
      size_t closest = closest_root(attractor[y * xsz + x]);
      for (size_t c = 0; c < 3; c++) {
        (*output)[y * xsz * 3 + x * 3 + c] =
            colours[closest][c] * color_scale[y * xsz + x];
      }
    }
  }
}

int main() {
  size_t N;
  scanf("%lu", &N);
  if (N < 1) {
    return 2;
  }
  std::vector<double> poly(N + 1);
  for (size_t i = 0; i < N + 1; i++) {
    scanf("%lf", &poly[N - i]);
  }
  std::vector<std::array<int, 3>> colours(N);
  for (size_t i = 0; i < N; i++) {
    scanf("%d%d%d", &colours[i][0], &colours[i][1], &colours[i][2]);
  }
  float scale; // min/max coord on shortest axis.
  scanf("%f", &scale);
  size_t min_sz;
  scanf("%lu", &min_sz);
  size_t pad_x, pad_y;
  scanf("%lu%lu", &pad_x, &pad_y);
  std::vector<std::string> pattern;
  bool has_px = false;
  while (true) {
    std::string line;
    std::cin >> line;
    if (!pattern.empty() && pattern.back().size() != line.size()) {
      break;
    }
    pattern.push_back(line);
    for (char c : line) {
      if (c == '#')
        has_px = true;
    }
  }
  if (!has_px) {
    return 1;
  }
  size_t pxsz = pattern[0].size() + 2 * pad_x;
  size_t pysz = pattern.size() + 2 * pad_y;

  size_t scaling = 0;
  if (pxsz < pysz) {
    scaling = (min_sz + pxsz - 1) / pxsz;
  } else {
    scaling = (min_sz + pysz - 1) / pysz;
  }
  size_t xsz = scaling * pxsz;
  size_t ysz = scaling * pysz;
  PatternT pattern_img;
  ComputePattern(&pattern_img, pattern, xsz, ysz, pad_x, pad_y, pxsz, pysz,
                 scale);
  std::vector<unsigned char> output(xsz * ysz * 3);
  ComputeFractal(pattern_img, &output, xsz, ysz, scale, poly, colours);
  WritePNG(output.data(), xsz, ysz, "attractor.png");
}
