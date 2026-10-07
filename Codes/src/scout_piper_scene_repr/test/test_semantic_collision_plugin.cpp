// CollisionEnvSemantic with a real MoveIt robot model and a synthetic field.
//
// A box-shaped probe on two prismatic joints (x, y) moves around a vertical
// stem described by a SemanticDistanceField published on the listener's topic.

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <thread>

#include <moveit/collision_detection/collision_common.h>
#include <moveit/collision_detection/collision_matrix.h>
#include <moveit/collision_detection/collision_plugin_cache.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <rclcpp/rclcpp.hpp>
#include <srdfdom/model.h>
#include <urdf_parser/urdf_parser.h>

#include "scout_piper_scene_repr/msg/semantic_distance_field.hpp"
#include "scout_piper_scene_repr/semantic_collision_plugin.hpp"
#include "scout_piper_scene_repr/semantic_field_listener.hpp"

namespace cd = collision_detection;
using scout_piper_scene_repr::CollisionEnvSemantic;
using scout_piper_scene_repr::SemanticFieldListener;

namespace
{

constexpr double kStemX = 0.40, kStemY = 0.0, kStemR = 0.005, kStemPad = 0.005;
constexpr double kVoxel = 0.01;
constexpr double kUnknownBeyondX = 0.55;   // voxels with x > this were never observed

const char * kUrdf = R"(<?xml version="1.0"?>
<robot name="probe_robot">
  <link name="base"/>
  <link name="carriage"/>
  <link name="probe">
    <collision><geometry><box size="0.02 0.02 0.1"/></geometry></collision>
  </link>
  <joint name="slide_x" type="prismatic">
    <parent link="base"/><child link="carriage"/><axis xyz="1 0 0"/>
    <limit lower="-1.0" upper="1.0" effort="1" velocity="1"/>
  </joint>
  <joint name="slide_y" type="prismatic">
    <parent link="carriage"/><child link="probe"/><axis xyz="0 1 0"/>
    <origin xyz="0 0 0.2"/>
    <limit lower="-1.0" upper="1.0" effort="1" velocity="1"/>
  </joint>
</robot>)";

const char * kSrdf = R"(<?xml version="1.0"?>
<robot name="probe_robot">
  <group name="probe_group"><joint name="slide_x"/><joint name="slide_y"/></group>
</robot>)";

moveit::core::RobotModelPtr makeModel()
{
  auto urdf = urdf::parseURDF(kUrdf);
  auto srdf = std::make_shared<srdf::Model>();
  srdf->initString(*urdf, kSrdf);
  return std::make_shared<moveit::core::RobotModel>(urdf, srdf);
}

/// Stem along z at (kStemX, kStemY); grid 0.2..0.7 x -0.2..0.2 x 0..0.4 m.
scout_piper_scene_repr::msg::SemanticDistanceField makeField(
  const std::string & frame, const rclcpp::Time & stamp)
{
  scout_piper_scene_repr::msg::SemanticDistanceField m;
  m.header.frame_id = frame;
  m.header.stamp = stamp;
  m.origin.x = 0.2;
  m.origin.y = -0.2;
  m.origin.z = 0.0;
  m.voxel_size = kVoxel;
  m.size = {50, 40, 40};
  m.hard_classes = {"stem", "branch", "other"};
  const std::size_t n = 50 * 40 * 40;
  m.hard_distance.resize(n);
  m.hard_class.assign(n, 0);
  m.age_ds.assign(n, 0);
  for (std::uint32_t i = 0; i < 50; ++i) {
    for (std::uint32_t j = 0; j < 40; ++j) {
      for (std::uint32_t k = 0; k < 40; ++k) {
        const double x = 0.2 + (i + 0.5) * kVoxel, y = -0.2 + (j + 0.5) * kVoxel;
        const std::size_t f = (static_cast<std::size_t>(i) * 40 + j) * 40 + k;
        m.hard_distance[f] = static_cast<float>(std::hypot(x - kStemX, y - kStemY) - kStemR - kStemPad);
        if (x > kUnknownBeyondX) {
          m.age_ds[f] = 65535;
        }
      }
    }
  }
  return m;
}

class SemanticPluginTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    model_ = makeModel();
    env_ = std::make_shared<CollisionEnvSemantic>(model_);   // starts the listener
  }

  static void TearDownTestSuite()
  {
    env_.reset();
    model_.reset();
  }

  moveit::core::RobotState at(double x, double y = 0.0) const
  {
    moveit::core::RobotState s(model_);
    s.setToDefaultValues();
    s.setVariablePosition("slide_x", x);
    s.setVariablePosition("slide_y", y);
    s.update();
    return s;
  }

  bool collides(const moveit::core::RobotState & s, cd::CollisionResult * out = nullptr) const
  {
    cd::CollisionRequest req;
    req.contacts = out != nullptr;
    req.max_contacts = 10;
    cd::CollisionResult res;
    env_->checkRobotCollision(req, res, s);
    if (out) {
      *out = res;
    }
    return res.collision;
  }

  static void publishField()
  {
    static auto pub_node = std::make_shared<rclcpp::Node>("field_test_publisher");
    static auto pub = pub_node->create_publisher<scout_piper_scene_repr::msg::SemanticDistanceField>(
      "/scene_repr/distance_field", rclcpp::QoS(1).reliable().transient_local());
    pub->publish(makeField(model_->getModelFrame(), pub_node->now()));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!SemanticFieldListener::instance().snapshot().reason.empty() &&
      std::chrono::steady_clock::now() < deadline)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    ASSERT_EQ(SemanticFieldListener::instance().snapshot().reason, "");
  }

  static moveit::core::RobotModelPtr model_;
  static std::shared_ptr<CollisionEnvSemantic> env_;
};

moveit::core::RobotModelPtr SemanticPluginTest::model_;
std::shared_ptr<CollisionEnvSemantic> SemanticPluginTest::env_;

}  // namespace

// Order matters: the field is published by the second test.
TEST_F(SemanticPluginTest, a_without_field_every_check_collides)
{
  EXPECT_NE(SemanticFieldListener::instance().snapshot().reason, "");
  EXPECT_TRUE(collides(at(0.0)));
  // self-collision stays FCL's and is unaffected by the field
  cd::CollisionRequest req;
  cd::CollisionResult res;
  env_->checkSelfCollision(req, res, at(0.0));
  EXPECT_FALSE(res.collision);
}

TEST_F(SemanticPluginTest, b_field_arrives)
{
  publishField();
  EXPECT_EQ(scout_piper_scene_repr::computeRobotSpheres(model_).size(), 1u);
  EXPECT_EQ(scout_piper_scene_repr::computeRobotSpheres(model_).front().spheres.size(), 8u);
}

TEST_F(SemanticPluginTest, c_free_near_and_inside_the_stem)
{
  // probe cover: 8 spheres of r ≈ 0.0155 m on the box axis; stem + padding = 0.01 m
  EXPECT_FALSE(collides(at(kStemX - 0.05)));
  EXPECT_FALSE(collides(at(kStemX + 0.05)));
  cd::CollisionResult res;
  EXPECT_TRUE(collides(at(kStemX - 0.015), &res));
  ASSERT_EQ(res.contacts.count({"probe", "semantic/stem"}), 1u);
  EXPECT_GT(res.contacts.at({"probe", "semantic/stem"}).front().depth, 0.0);
  EXPECT_TRUE(collides(at(kStemX)));
}

TEST_F(SemanticPluginTest, d_unknown_and_outside)
{
  EXPECT_TRUE(collides(at(kUnknownBeyondX + 0.05)));   // unknown space is an obstacle
  EXPECT_FALSE(collides(at(-0.3)));                     // outside the grid: field is silent
}

TEST_F(SemanticPluginTest, e_acm_exempts_the_link)
{
  cd::AllowedCollisionMatrix acm;
  acm.setEntry("probe", scout_piper_scene_repr::kSemanticBody, true);
  cd::CollisionRequest req;
  cd::CollisionResult res;
  env_->checkRobotCollision(req, res, at(kStemX), acm);
  EXPECT_FALSE(res.collision);
}

TEST_F(SemanticPluginTest, f_distance_matches_geometry)
{
  cd::DistanceRequest req;
  req.enable_nearest_points = true;
  cd::DistanceResult res;
  env_->distanceRobot(req, res, at(kStemX - 0.05));
  const double sphere_r = std::sqrt(2 * 0.01 * 0.01 + 0.25 * 0.0125 * 0.0125);
  EXPECT_NEAR(res.minimum_distance.distance, 0.05 - kStemR - kStemPad - sphere_r, 0.002);
  EXPECT_EQ(res.minimum_distance.link_names[1], "semantic/stem");
  EXPECT_LT(res.minimum_distance.normal.x(), -0.9);   // points from the stem to the probe
  EXPECT_FALSE(res.collision);
}

TEST_F(SemanticPluginTest, g_continuous_check_catches_the_stem_between_two_free_states)
{
  const auto a = at(kStemX - 0.08), b = at(kStemX + 0.08);
  EXPECT_FALSE(collides(a));
  EXPECT_FALSE(collides(b));
  cd::CollisionRequest req;
  cd::CollisionResult res;
  env_->checkRobotCollision(req, res, a, b);
  EXPECT_TRUE(res.collision);
}

// The path move_group takes for `collision_detector: "Semantic"`: pluginlib
// lookup by name, then the plugin installs the allocator on the scene. Needs
// the package's install space on the ament index (colcon test sources it).
TEST_F(SemanticPluginTest, h_move_group_loads_the_plugin_by_name)
{
  auto scene = std::make_shared<planning_scene::PlanningScene>(model_);
  collision_detection::CollisionPluginCache cache;
  ASSERT_TRUE(cache.activate("Semantic", scene));
  EXPECT_EQ(scene->getCollisionDetectorName(), "Semantic");
  auto & state = scene->getCurrentStateNonConst();
  state.setVariablePosition("slide_x", kStemX);
  state.update();
  EXPECT_TRUE(scene->isStateColliding(state, "", false));
  state.setVariablePosition("slide_x", kStemX - 0.05);
  state.update();
  EXPECT_FALSE(scene->isStateColliding(state, "", false));
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  const int ret = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return ret;
}
