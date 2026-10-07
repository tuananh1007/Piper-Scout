// Test driver for semantic_distance_field.hpp, run by test_field_snapshot.py.
//
//   field_query_driver FIELD_BIN QUERY_BIN
//   field_query_driver --cover      sphere-cover self-check, prints "ok" on success
//
// FIELD_BIN (little endian): f64 origin[3], f64 voxel_size, u32 size[3], f64 stamp,
//   f64 soft_max_penetration, u8 has_soft, then f32 hard[N], u8 hard_class[N],
//   [f32 soft[N]], u16 age_ds[N].
// QUERY_BIN: u32 M, f64 now, f64 max_voxel_age_s, u8 unknown_is_occupied,
//   u8 check_soft, then M x f64 (x, y, z, radius).
// Prints one line per query: in_bounds known fresh hard hard_class soft status clearance collision.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "scout_piper_scene_repr/semantic_distance_field.hpp"

using scout_piper_scene_repr::SemanticDistanceField;

template<class T>
T get(std::ifstream & f)
{
  T v{};
  f.read(reinterpret_cast<char *>(&v), sizeof(T));
  if (!f) {throw std::runtime_error("short read");}
  return v;
}

template<class T>
std::vector<T> getv(std::ifstream & f, std::size_t n)
{
  std::vector<T> v(n);
  f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
  if (!f) {throw std::runtime_error("short read");}
  return v;
}

// Boxes of several aspect ratios: every point of the box must lie inside one of
// the covering spheres, and the spheres must not be much larger than needed.
int coverCheck()
{
  using scout_piper_scene_repr::Vec3;
  const std::vector<Vec3> dims{{0.02, 0.02, 0.30}, {0.10, 0.04, 0.04}, {0.05, 0.05, 0.05},
    {0.30, 0.01, 0.08}, {0.0, 0.0, 0.0}};
  for (const auto & d : dims) {
    std::vector<Vec3> corners;
    for (int c = 0; c < 8; ++c) {
      corners.push_back({(c & 1 ? 0.5 : -0.5) * d[0] + 0.1, (c & 2 ? 0.5 : -0.5) * d[1] - 0.2,
          (c & 4 ? 0.5 : -0.5) * d[2] + 0.3});
    }
    const auto cyl = scout_piper_scene_repr::fitBoundingCylinder(corners);
    const auto spheres = scout_piper_scene_repr::spheresCoveringCylinder(cyl, 0.0);
    const int g = 12;
    for (int i = 0; i <= g; ++i) {
      for (int j = 0; j <= g; ++j) {
        for (int k = 0; k <= g; ++k) {
          const Vec3 p{corners[0][0] + d[0] * i / g, corners[0][1] + d[1] * j / g,
            corners[0][2] + d[2] * k / g};
          bool inside = false;
          for (const auto & s : spheres) {
            const double dx = p[0] - s.center[0], dy = p[1] - s.center[1], dz = p[2] - s.center[2];
            inside = inside || dx * dx + dy * dy + dz * dz <= s.radius * s.radius + 1e-12;
          }
          if (!inside) {
            std::printf("uncovered point for box %g %g %g\n", d[0], d[1], d[2]);
            return 1;
          }
        }
      }
    }
    const double box_half_diag = 0.5 * std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    for (const auto & s : spheres) {
      if (s.radius > 1.2 * std::max(cyl.radius, 1e-4) + 1e-12 || s.radius > box_half_diag + 1e-3) {
        std::printf("sphere too large for box %g %g %g: %g\n", d[0], d[1], d[2], s.radius);
        return 1;
      }
    }
  }
  std::printf("ok\n");
  return 0;
}

int main(int argc, char ** argv)
{
  if (argc == 2 && std::string(argv[1]) == "--cover") {
    return coverCheck();
  }
  if (argc != 3) {
    std::cerr << "usage: field_query_driver FIELD_BIN QUERY_BIN\n";
    return 2;
  }
  std::ifstream ff(argv[1], std::ios::binary), qf(argv[2], std::ios::binary);
  SemanticDistanceField fld;
  for (auto & o : fld.origin) {o = get<double>(ff);}
  fld.voxel_size = get<double>(ff);
  for (auto & s : fld.size) {s = get<std::uint32_t>(ff);}
  fld.stamp = get<double>(ff);
  fld.soft_max_penetration = get<double>(ff);
  const bool has_soft = get<std::uint8_t>(ff) != 0;
  const std::size_t n = fld.count();
  fld.hard_distance = getv<float>(ff, n);
  fld.hard_class = getv<std::uint8_t>(ff, n);
  if (has_soft) {fld.soft_distance = getv<float>(ff, n);}
  fld.age_ds = getv<std::uint16_t>(ff, n);
  const std::string err = fld.finalize();
  if (!err.empty()) {
    std::cerr << "invalid field: " << err << "\n";
    return 3;
  }

  const auto m = get<std::uint32_t>(qf);
  scout_piper_scene_repr::SphereCheckConfig cfg;
  const double now = get<double>(qf);
  cfg.max_voxel_age_s = get<double>(qf);
  cfg.unknown_is_occupied = get<std::uint8_t>(qf) != 0;
  cfg.check_soft = get<std::uint8_t>(qf) != 0;
  for (std::uint32_t i = 0; i < m; ++i) {
    scout_piper_scene_repr::Vec3 p{get<double>(qf), get<double>(qf), get<double>(qf)};
    const double r = get<double>(qf);
    const auto q = fld.query(p, now, cfg.max_voxel_age_s);
    const auto s = fld.checkSphere(p, r, now, cfg);
    std::printf("%d %d %d %.12g %d %.12g %s %.12g %d\n", q.in_bounds, q.known, q.fresh, q.hard,
      q.hard_class, q.soft, scout_piper_scene_repr::toString(s.status), s.clearance, s.collision);
  }
  return 0;
}
