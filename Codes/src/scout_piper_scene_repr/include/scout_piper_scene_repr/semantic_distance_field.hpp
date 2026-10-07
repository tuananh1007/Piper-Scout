#pragma once

// Semantic distance field snapshot + sphere checks (P1.7.7).
//
// C++ twin of python/scout_piper_scene_repr_py/field.py (FieldSampler), which
// is itself checked against SemanticDistanceQuery. The data come from
// msg/SemanticDistanceField.msg, published by scene_query_node.
//
// Header-only and free of ROS / MoveIt / Eigen so it can be unit-tested with a
// plain compiler (test/field_query_driver.cpp, driven by
// test/test_field_snapshot.py).
//
// Conventions (same as the Python query):
//   * voxel (i, j, k) spans origin + [i, i+1) * voxel_size; values live at
//     voxel centres; arrays are C-order over (i, j, k), k fastest;
//   * signed distance: > 0 outside, < 0 inside, metres; trilinear
//     interpolation over voxel centres with coordinates clamped to the grid;
//   * a grid that is +inf everywhere means "no voxel of this kind"; otherwise
//     +inf entries are replaced by 1e3 m before interpolation;
//   * unknown (never observed) or stale voxels are never reported free.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace scout_piper_scene_repr
{

using Vec3 = std::array<double, 3>;

constexpr std::uint16_t kAgeUnknown = 65535;   // age_ds of a never-observed voxel
constexpr double kAgePerSecond = 10.0;         // age_ds units per second
constexpr double kInfSubstitute = 1e3;

struct FieldPoint
{
  bool in_bounds{false};
  bool known{false};
  bool fresh{false};
  double hard{std::numeric_limits<double>::infinity()};   // min over hard classes of (d - padding)
  int hard_class{-1};                                     // index into hard_classes, -1 if none
  double soft{std::numeric_limits<double>::infinity()};   // soft-class (leaf) signed distance
};

enum class SphereStatus
{
  Outside,   // centre outside the grid: the field says nothing
  Free,
  Hard,      // sphere reaches into a hard class (padding included)
  Soft,      // sphere pushes deeper into the soft class than soft_max_penetration
  Unknown,   // centre voxel never observed
  Stale,     // centre voxel older than max_voxel_age_s
};

inline const char * toString(SphereStatus s)
{
  switch (s) {
    case SphereStatus::Outside: return "outside";
    case SphereStatus::Free: return "free";
    case SphereStatus::Hard: return "hard";
    case SphereStatus::Soft: return "soft";
    case SphereStatus::Unknown: return "unknown";
    case SphereStatus::Stale: return "stale";
  }
  return "?";
}

struct SphereCheckConfig
{
  bool unknown_is_occupied{true};   // unknown / stale voxels count as collision
  bool check_soft{true};            // apply the soft-class penetration cap
  double max_voxel_age_s{30.0};
};

struct SphereResult
{
  SphereStatus status{SphereStatus::Outside};
  double clearance{std::numeric_limits<double>::infinity()};   // < 0 means penetration
  bool collision{false};
  int hard_class{-1};
  bool nearest_is_soft{false};   // clearance comes from the soft class, not a hard one
};

/// Cylinder in its own frame: axis = local z, centred at the origin.
struct BoundingCylinder
{
  Vec3 center{0.0, 0.0, 0.0};   // in the shape frame
  int axis{2};                  // shape-frame axis the cylinder runs along (0, 1, 2)
  double radius{0.0};
  double length{0.0};
};

struct Sphere
{
  Vec3 center{0.0, 0.0, 0.0};   // in the shape frame
  double radius{0.0};
};

/// Fit an axis-aligned bounding cylinder to points in the shape frame: the axis
/// is the longest bounding-box axis, the radius the largest distance from it.
inline BoundingCylinder fitBoundingCylinder(const std::vector<Vec3> & pts)
{
  BoundingCylinder c;
  if (pts.empty()) {
    return c;
  }
  Vec3 lo = pts.front(), hi = pts.front();
  for (const auto & p : pts) {
    for (int a = 0; a < 3; ++a) {
      lo[a] = std::min(lo[a], p[a]);
      hi[a] = std::max(hi[a], p[a]);
    }
  }
  int axis = 0;
  for (int a = 1; a < 3; ++a) {
    if (hi[a] - lo[a] > hi[axis] - lo[axis]) {
      axis = a;
    }
  }
  for (int a = 0; a < 3; ++a) {
    c.center[a] = 0.5 * (lo[a] + hi[a]);
  }
  c.axis = axis;
  c.length = hi[axis] - lo[axis];
  const int u = (axis + 1) % 3, v = (axis + 2) % 3;
  double r2 = 0.0;
  for (const auto & p : pts) {
    const double du = p[u] - c.center[u], dv = p[v] - c.center[v];
    r2 = std::max(r2, du * du + dv * dv);
  }
  c.radius = std::sqrt(r2);
  return c;
}

/// Spheres whose union covers the cylinder: n = ceil(L / r) equal segments,
/// one sphere per segment with radius sqrt(r^2 + (s/2)^2) (≤ 1.12 r).
inline std::vector<Sphere> spheresCoveringCylinder(const BoundingCylinder & c, double padding)
{
  std::vector<Sphere> out;
  const double r = std::max(c.radius, 1e-4);
  const int n = std::max(1, static_cast<int>(std::ceil(c.length / r - 1e-9)));
  const double s = c.length / n;
  const double R = std::sqrt(r * r + 0.25 * s * s) + padding;
  for (int i = 0; i < n; ++i) {
    Sphere sp;
    sp.center = c.center;
    sp.center[c.axis] += -0.5 * c.length + (i + 0.5) * s;
    sp.radius = R;
    out.push_back(sp);
  }
  return out;
}

class SemanticDistanceField
{
public:
  // ---- data (filled from the message, then call finalize()) ----
  Vec3 origin{0.0, 0.0, 0.0};
  double voxel_size{0.0};
  std::array<std::uint32_t, 3> size{0, 0, 0};
  double stamp{0.0};                       // seconds; ages refer to this time
  std::string frame_id;
  std::vector<std::string> hard_classes;
  std::vector<float> hard_distance;        // nx*ny*nz
  std::vector<std::uint8_t> hard_class;    // nx*ny*nz
  std::string soft_class;
  double soft_max_penetration{0.0};
  std::vector<float> soft_distance;        // nx*ny*nz or empty
  std::vector<std::uint16_t> age_ds;       // nx*ny*nz

  /// Check sizes and precompute interpolation grids. Returns "" when valid,
  /// otherwise a human-readable reason (and the field must not be used).
  std::string finalize()
  {
    const std::size_t n = count();
    if (voxel_size <= 0.0 || n == 0) {
      return "empty grid or non-positive voxel size";
    }
    if (hard_distance.size() != n || hard_class.size() != n || age_ds.size() != n) {
      return "array sizes do not match size[]";
    }
    if (!soft_distance.empty() && soft_distance.size() != n) {
      return "soft_distance size does not match size[]";
    }
    hard_ = prepare(hard_distance);
    soft_ = prepare(soft_distance);
    has_hard_ = !hard_.empty();
    has_soft_ = !soft_.empty();
    return "";
  }

  std::size_t count() const
  {
    return static_cast<std::size_t>(size[0]) * size[1] * size[2];
  }

  bool hasHard() const {return has_hard_;}
  bool hasSoft() const {return has_soft_;}

  FieldPoint query(const Vec3 & p, double now, double max_voxel_age_s) const
  {
    FieldPoint q;
    std::array<long, 3> idx{};
    bool inb = true;
    for (int a = 0; a < 3; ++a) {
      idx[a] = static_cast<long>(std::floor((p[a] - origin[a]) / voxel_size));
      inb = inb && idx[a] >= 0 && idx[a] < static_cast<long>(size[a]);
      idx[a] = std::clamp(idx[a], 0L, static_cast<long>(size[a]) - 1);
    }
    const std::size_t flat = linear(idx[0], idx[1], idx[2]);
    const std::uint16_t age = age_ds[flat];
    q.in_bounds = inb;
    q.known = inb && age != kAgeUnknown;
    q.fresh = q.known && (age / kAgePerSecond + (now - stamp) <= max_voxel_age_s);
    q.hard = has_hard_ ? interpolate(hard_, p) : std::numeric_limits<double>::infinity();
    q.hard_class = has_hard_ ? static_cast<int>(hard_class[flat]) : -1;
    q.soft = has_soft_ ? interpolate(soft_, p) : std::numeric_limits<double>::infinity();
    return q;
  }

  /// Central-difference gradient of the hard field (points away from obstacles).
  Vec3 hardGradient(const Vec3 & p) const
  {
    Vec3 g{0.0, 0.0, 0.0};
    if (!has_hard_) {
      return g;
    }
    const double eps = voxel_size;
    for (int a = 0; a < 3; ++a) {
      Vec3 lo = p, hi = p;
      lo[a] -= eps;
      hi[a] += eps;
      g[a] = (interpolate(hard_, hi) - interpolate(hard_, lo)) / (2.0 * eps);
    }
    return g;
  }

  /// Sphere check; mirrors FieldSampler.check_spheres in field.py.
  SphereResult checkSphere(
    const Vec3 & center, double radius, double now, const SphereCheckConfig & cfg) const
  {
    SphereResult r;
    const FieldPoint q = query(center, now, cfg.max_voxel_age_s);
    r.hard_class = q.hard_class;
    if (!q.in_bounds) {
      return r;   // Outside, no collision
    }
    if (!q.fresh && cfg.unknown_is_occupied) {
      r.status = q.known ? SphereStatus::Stale : SphereStatus::Unknown;
      r.clearance = std::min(q.hard, 0.0) - radius;
      r.collision = true;
      return r;
    }
    const double clear_hard = q.hard - radius;
    const double clear_soft = cfg.check_soft ?
      q.soft - radius + soft_max_penetration : std::numeric_limits<double>::infinity();
    r.nearest_is_soft = clear_soft < clear_hard;
    if (clear_hard < 0.0) {
      r.status = SphereStatus::Hard;
      r.clearance = std::min(clear_hard, clear_soft);
      r.collision = true;
    } else if (clear_soft < 0.0) {
      r.status = SphereStatus::Soft;
      r.clearance = clear_soft;
      r.collision = true;
    } else {
      r.status = SphereStatus::Free;
      r.clearance = std::min(clear_hard, clear_soft);
    }
    return r;
  }

private:
  std::vector<double> hard_, soft_;
  bool has_hard_{false}, has_soft_{false};

  std::size_t linear(long i, long j, long k) const
  {
    return (static_cast<std::size_t>(i) * size[1] + static_cast<std::size_t>(j)) * size[2] +
           static_cast<std::size_t>(k);
  }

  static std::vector<double> prepare(const std::vector<float> & grid)
  {
    bool all_inf = true;
    for (float v : grid) {
      if (!std::isinf(v)) {
        all_inf = false;
        break;
      }
    }
    if (grid.empty() || all_inf) {
      return {};
    }
    std::vector<double> out(grid.size());
    for (std::size_t i = 0; i < grid.size(); ++i) {
      out[i] = std::isinf(grid[i]) ? kInfSubstitute : static_cast<double>(grid[i]);
    }
    return out;
  }

  double interpolate(const std::vector<double> & grid, const Vec3 & p) const
  {
    std::array<long, 3> i0{}, i1{};
    std::array<double, 3> t{};
    for (int a = 0; a < 3; ++a) {
      const long n = static_cast<long>(size[a]);
      const double g = (p[a] - origin[a]) / voxel_size - 0.5;
      const double gc = std::clamp(g, 0.0, static_cast<double>(n - 1));
      i0[a] = std::min(static_cast<long>(std::floor(gc)), std::max(n - 2, 0L));
      t[a] = gc - static_cast<double>(i0[a]);
      i1[a] = std::min(i0[a] + 1, n - 1);
    }
    double out = 0.0;
    for (int dx = 0; dx < 2; ++dx) {
      const double wx = dx ? t[0] : 1.0 - t[0];
      const long ix = dx ? i1[0] : i0[0];
      for (int dy = 0; dy < 2; ++dy) {
        const double wy = dy ? t[1] : 1.0 - t[1];
        const long iy = dy ? i1[1] : i0[1];
        for (int dz = 0; dz < 2; ++dz) {
          const double wz = dz ? t[2] : 1.0 - t[2];
          const long iz = dz ? i1[2] : i0[2];
          out += wx * wy * wz * grid[linear(ix, iy, iz)];
        }
      }
    }
    return out;
  }
};

}  // namespace scout_piper_scene_repr
