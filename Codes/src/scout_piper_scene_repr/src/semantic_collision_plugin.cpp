// Semantic collision detector: FCL + semantic distance field (P1.3.1 / P1.7.7).
//
// Registered with pluginlib as a collision_detection::CollisionPlugin named
// "Semantic"; select it with `collision_detector: "Semantic"` in the move_group
// configuration. See semantic_collision_plugin.hpp for the
// behaviour and docs/PHASE1_RUNTIME.md for how to run it.

#include "scout_piper_scene_repr/semantic_collision_plugin.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include <geometric_shapes/shapes.h>
#include <moveit/planning_scene/planning_scene.h>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>

#include "scout_piper_scene_repr/semantic_field_listener.hpp"

namespace scout_piper_scene_repr
{

namespace
{

namespace cd = collision_detection;

rclcpp::Clock & steadyClock()
{
  static rclcpp::Clock clock(RCL_STEADY_TIME);
  return clock;
}

/// "semantic/<class>" of the obstacle that sets the sphere's clearance.
std::string fieldBodyName(const SemanticDistanceField & field, const SphereResult & r)
{
  switch (r.status) {
    case SphereStatus::Unknown:
      return kSemanticBody + "/unknown";
    case SphereStatus::Stale:
      return kSemanticBody + "/stale";
    case SphereStatus::Soft:
      return kSemanticBody + "/" + field.soft_class;
    case SphereStatus::Hard:
    case SphereStatus::Free:
      if (r.status == SphereStatus::Free && r.nearest_is_soft) {
        return kSemanticBody + "/" + field.soft_class;
      }
      if (r.hard_class >= 0 && static_cast<std::size_t>(r.hard_class) < field.hard_classes.size()) {
        return kSemanticBody + "/" + field.hard_classes[r.hard_class];
      }
      return kSemanticBody + "/hard";
    default:
      return kSemanticBody;
  }
}

/// Unit normal (model frame) pointing from the obstacle towards the sphere.
Eigen::Vector3d fieldNormal(
  const SemanticDistanceField & field, const Eigen::Vector3d & c_field,
  const Eigen::Isometry3d & model_T_field)
{
  const Vec3 g = field.hardGradient({c_field.x(), c_field.y(), c_field.z()});
  Eigen::Vector3d n(g[0], g[1], g[2]);
  if (n.norm() < 1e-9) {
    n = Eigen::Vector3d::UnitZ();
  }
  return model_T_field.linear() * n.normalized();
}

void addContact(
  const cd::CollisionRequest & req, cd::CollisionResult & res,
  const SemanticDistanceField & field, const std::string & link, const SphereResult & r,
  const Eigen::Vector3d & c_field, double radius, const Eigen::Isometry3d & model_T_field)
{
  if (res.contact_count >= req.max_contacts) {
    return;
  }
  const std::string other = fieldBodyName(field, r);
  auto & list = res.contacts[std::make_pair(link, other)];
  if (list.size() >= req.max_contacts_per_pair) {
    return;
  }
  const Eigen::Vector3d n = fieldNormal(field, c_field, model_T_field);
  const Eigen::Vector3d c = model_T_field * c_field;
  cd::Contact contact;
  contact.pos = c - radius * n;
  contact.normal = n;
  contact.depth = -r.clearance;
  contact.body_name_1 = link;
  contact.body_type_1 = cd::BodyTypes::ROBOT_LINK;
  contact.body_name_2 = other;
  contact.body_type_2 = cd::BodyTypes::WORLD_OBJECT;
  contact.percent_interpolation = 0.0;
  contact.nearest_points[0] = contact.pos;
  contact.nearest_points[1] = c - (radius + r.clearance) * n;
  list.push_back(contact);
  ++res.contact_count;
}

std::vector<Sphere> shapeSpheres(const shapes::Shape & shape)
{
  switch (shape.type) {
    case shapes::SPHERE:
      return {Sphere{{0.0, 0.0, 0.0}, static_cast<const shapes::Sphere &>(shape).radius}};
    case shapes::CYLINDER: {
        const auto & c = static_cast<const shapes::Cylinder &>(shape);
        BoundingCylinder b;
        b.radius = c.radius;
        b.length = c.length;
        return spheresCoveringCylinder(b, 0.0);
      }
    case shapes::CONE: {
        const auto & c = static_cast<const shapes::Cone &>(shape);
        BoundingCylinder b;
        b.radius = c.radius;
        b.length = c.length;
        return spheresCoveringCylinder(b, 0.0);
      }
    case shapes::BOX: {
        const auto & box = static_cast<const shapes::Box &>(shape);
        std::vector<Vec3> corners;
        for (int i = 0; i < 8; ++i) {
          corners.push_back(
            {(i & 1 ? 0.5 : -0.5) * box.size[0], (i & 2 ? 0.5 : -0.5) * box.size[1],
              (i & 4 ? 0.5 : -0.5) * box.size[2]});
        }
        return spheresCoveringCylinder(fitBoundingCylinder(corners), 0.0);
      }
    case shapes::MESH: {
        const auto & mesh = static_cast<const shapes::Mesh &>(shape);
        std::vector<Vec3> pts;
        pts.reserve(mesh.vertex_count);
        for (unsigned int v = 0; v < mesh.vertex_count; ++v) {
          pts.push_back({mesh.vertices[3 * v], mesh.vertices[3 * v + 1], mesh.vertices[3 * v + 2]});
        }
        if (pts.empty()) {
          return {};
        }
        return spheresCoveringCylinder(fitBoundingCylinder(pts), 0.0);
      }
    default:
      return {};   // planes, octrees: not robot geometry
  }
}

}  // namespace

std::vector<LinkShapeSpheres> computeRobotSpheres(const moveit::core::RobotModelConstPtr & model)
{
  std::vector<LinkShapeSpheres> out;
  for (const moveit::core::LinkModel * link : model->getLinkModelsWithCollisionGeometry()) {
    const auto & link_shapes = link->getShapes();
    for (std::size_t i = 0; i < link_shapes.size(); ++i) {
      if (!link_shapes[i]) {
        continue;
      }
      LinkShapeSpheres ls;
      ls.link = link;
      ls.shape_index = i;
      ls.spheres = shapeSpheres(*link_shapes[i]);
      if (!ls.spheres.empty()) {
        out.push_back(std::move(ls));
      }
    }
  }
  return out;
}

const std::string SemanticCollisionDetectorAllocator::NAME = "Semantic";

bool SemanticCollisionPluginLoader::initialize(const planning_scene::PlanningScenePtr & scene) const
{
  scene->allocateCollisionDetector(SemanticCollisionDetectorAllocator::create());
  return true;
}

CollisionEnvSemantic::CollisionEnvSemantic(
  const moveit::core::RobotModelConstPtr & robot_model, double padding, double scale)
: cd::CollisionEnvFCL(robot_model, padding, scale)
, spheres_(std::make_shared<const std::vector<LinkShapeSpheres>>(computeRobotSpheres(robot_model)))
{
  SemanticFieldListener::instance().registerModelFrame(robot_model->getModelFrame());
}

CollisionEnvSemantic::CollisionEnvSemantic(
  const moveit::core::RobotModelConstPtr & robot_model, const cd::WorldPtr & world,
  double padding, double scale)
: cd::CollisionEnvFCL(robot_model, world, padding, scale)
, spheres_(std::make_shared<const std::vector<LinkShapeSpheres>>(computeRobotSpheres(robot_model)))
{
  SemanticFieldListener::instance().registerModelFrame(robot_model->getModelFrame());
}

CollisionEnvSemantic::CollisionEnvSemantic(
  const CollisionEnvSemantic & other, const cd::WorldPtr & world)
: cd::CollisionEnvFCL(other, world)
, spheres_(other.spheres_)
{
}

void CollisionEnvSemantic::checkRobotCollision(
  const cd::CollisionRequest & req, cd::CollisionResult & res,
  const moveit::core::RobotState & state) const
{
  checkDiscrete(req, res, state, nullptr);
}

void CollisionEnvSemantic::checkRobotCollision(
  const cd::CollisionRequest & req, cd::CollisionResult & res,
  const moveit::core::RobotState & state, const cd::AllowedCollisionMatrix & acm) const
{
  checkDiscrete(req, res, state, &acm);
}

void CollisionEnvSemantic::checkRobotCollision(
  const cd::CollisionRequest & req, cd::CollisionResult & res,
  const moveit::core::RobotState & state1, const moveit::core::RobotState & state2) const
{
  checkContinuous(req, res, state1, state2, nullptr);
}

void CollisionEnvSemantic::checkRobotCollision(
  const cd::CollisionRequest & req, cd::CollisionResult & res,
  const moveit::core::RobotState & state1, const moveit::core::RobotState & state2,
  const cd::AllowedCollisionMatrix & acm) const
{
  checkContinuous(req, res, state1, state2, &acm);
}

void CollisionEnvSemantic::checkDiscrete(
  const cd::CollisionRequest & req, cd::CollisionResult & res,
  const moveit::core::RobotState & state, const cd::AllowedCollisionMatrix * acm) const
{
  if (acm) {
    cd::CollisionEnvFCL::checkRobotCollision(req, res, state, *acm);
  } else {
    cd::CollisionEnvFCL::checkRobotCollision(req, res, state);
  }
  if (res.collision && !req.contacts && !req.distance) {
    return;
  }
  checkField(req, res, state, acm);
}

void CollisionEnvSemantic::checkContinuous(
  const cd::CollisionRequest & req, cd::CollisionResult & res,
  const moveit::core::RobotState & state1, const moveit::core::RobotState & state2,
  const cd::AllowedCollisionMatrix * acm) const
{
  const double step = std::max(SemanticFieldListener::instance().config().continuous_step, 1e-4);
  const int n = std::max(1, static_cast<int>(std::ceil(state1.distance(state2) / step)));
  moveit::core::RobotState s(state1);
  for (int i = 0; i <= n; ++i) {
    state1.interpolate(state2, static_cast<double>(i) / n, s);
    s.update();
    checkDiscrete(req, res, s, acm);
    if (res.collision && !req.contacts && !req.distance) {
      return;
    }
  }
}

bool CollisionEnvSemantic::exempt(
  const std::string & link, const cd::AllowedCollisionMatrix * acm) const
{
  const auto & ignore = SemanticFieldListener::instance().config().ignore_links;
  if (std::find(ignore.begin(), ignore.end(), link) != ignore.end()) {
    return true;
  }
  if (!acm) {
    return false;
  }
  cd::AllowedCollision::Type type;
  if (acm->getEntry(link, kSemanticBody, type)) {
    return type == cd::AllowedCollision::ALWAYS;
  }
  return (acm->getDefaultEntry(link, type) && type == cd::AllowedCollision::ALWAYS) ||
         (acm->getDefaultEntry(kSemanticBody, type) && type == cd::AllowedCollision::ALWAYS);
}

void CollisionEnvSemantic::checkField(
  const cd::CollisionRequest & req, cd::CollisionResult & res,
  const moveit::core::RobotState & state, const cd::AllowedCollisionMatrix * acm) const
{
  auto & listener = SemanticFieldListener::instance();
  const auto & cfg = listener.config();
  const FieldSnapshot snap = listener.snapshot();
  if (!snap.reason.empty()) {
    if (cfg.require_field) {
      res.collision = true;
      if (req.distance) {
        res.distance = std::min(res.distance, 0.0);
      }
      RCLCPP_ERROR_THROTTLE(
        listener.logger(), steadyClock(), 5000,
        "Semantic collision: %s; reporting collision because require_field is true.",
        snap.reason.c_str());
    }
    return;
  }
  const SemanticDistanceField & field = *snap.field;
  SphereCheckConfig sc;
  sc.unknown_is_occupied = cfg.unknown_is_occupied;
  sc.check_soft = cfg.check_soft;
  sc.max_voxel_age_s = cfg.max_voxel_age_s;
  const moveit::core::JointModelGroup * group = nullptr;
  if (!req.group_name.empty() && getRobotModel()->hasJointModelGroup(req.group_name)) {
    group = getRobotModel()->getJointModelGroup(req.group_name);
  }
  const Eigen::Isometry3d model_T_field = snap.field_T_model.inverse();

  for (const auto & ls : *spheres_) {
    const std::string & name = ls.link->getName();
    if ((group && !group->isLinkUpdated(name)) || exempt(name, acm)) {
      continue;
    }
    const Eigen::Isometry3d field_T_shape =
      snap.field_T_model * state.getCollisionBodyTransform(ls.link, ls.shape_index);
    const double pad = getLinkPadding(name) + cfg.sphere_padding_m;
    for (const Sphere & s : ls.spheres) {
      const Eigen::Vector3d c = field_T_shape * Eigen::Vector3d(s.center[0], s.center[1], s.center[2]);
      const double radius = s.radius + pad;
      const SphereResult r = field.checkSphere({c.x(), c.y(), c.z()}, radius, snap.now, sc);
      if (r.status == SphereStatus::Outside) {
        continue;
      }
      if (req.distance) {
        res.distance = std::min(res.distance, r.clearance);
      }
      if (!r.collision) {
        continue;
      }
      res.collision = true;
      if (req.contacts) {
        addContact(req, res, field, name, r, c, radius, model_T_field);
      }
      if (!req.distance && (!req.contacts || res.contact_count >= req.max_contacts)) {
        return;
      }
    }
  }
}

void CollisionEnvSemantic::distanceRobot(
  const cd::DistanceRequest & req, cd::DistanceResult & res,
  const moveit::core::RobotState & state) const
{
  cd::CollisionEnvFCL::distanceRobot(req, res, state);
  auto & listener = SemanticFieldListener::instance();
  const auto & cfg = listener.config();
  const FieldSnapshot snap = listener.snapshot();
  if (!snap.reason.empty()) {
    if (cfg.require_field) {
      res.collision = true;
      res.minimum_distance.distance = std::min(res.minimum_distance.distance, 0.0);
      RCLCPP_ERROR_THROTTLE(
        listener.logger(), steadyClock(), 5000,
        "Semantic distance: %s; reporting contact because require_field is true.",
        snap.reason.c_str());
    }
    return;
  }
  const SemanticDistanceField & field = *snap.field;
  SphereCheckConfig sc;
  sc.unknown_is_occupied = cfg.unknown_is_occupied;
  sc.check_soft = cfg.check_soft;
  sc.max_voxel_age_s = cfg.max_voxel_age_s;
  const Eigen::Isometry3d model_T_field = snap.field_T_model.inverse();

  for (const auto & ls : *spheres_) {
    const std::string & name = ls.link->getName();
    if ((req.active_components_only && !req.active_components_only->count(ls.link)) ||
      exempt(name, req.acm))
    {
      continue;
    }
    const Eigen::Isometry3d field_T_shape =
      snap.field_T_model * state.getCollisionBodyTransform(ls.link, ls.shape_index);
    const double pad = getLinkPadding(name) + cfg.sphere_padding_m;
    for (const Sphere & s : ls.spheres) {
      const Eigen::Vector3d c = field_T_shape * Eigen::Vector3d(s.center[0], s.center[1], s.center[2]);
      const double radius = s.radius + pad;
      const SphereResult r = field.checkSphere({c.x(), c.y(), c.z()}, radius, snap.now, sc);
      if (r.status == SphereStatus::Outside) {
        continue;
      }
      res.collision = res.collision || r.collision;
      auto & m = res.minimum_distance;
      if (r.clearance < m.distance) {
        const Eigen::Vector3d n = fieldNormal(field, c, model_T_field);
        const Eigen::Vector3d cm = model_T_field * c;
        m.distance = r.clearance;
        m.link_names[0] = name;
        m.link_names[1] = fieldBodyName(field, r);
        m.body_types[0] = cd::BodyTypes::ROBOT_LINK;
        m.body_types[1] = cd::BodyTypes::WORLD_OBJECT;
        m.normal = n;
        m.nearest_points[0] = cm - radius * n;
        m.nearest_points[1] = cm - (radius + r.clearance) * n;
      }
    }
  }
}

}  // namespace scout_piper_scene_repr

PLUGINLIB_EXPORT_CLASS(
  scout_piper_scene_repr::SemanticCollisionPluginLoader,
  collision_detection::CollisionPlugin)
