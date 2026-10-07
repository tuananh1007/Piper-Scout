#include "scout_piper_scene_repr/semantic_field_listener.hpp"

#include <chrono>
#include <utility>

#include <tf2/exceptions.h>

namespace scout_piper_scene_repr
{

using FieldMsg = scout_piper_scene_repr::msg::SemanticDistanceField;

SemanticFieldListener & SemanticFieldListener::instance()
{
  static SemanticFieldListener listener;
  return listener;
}

SemanticFieldListener::SemanticFieldListener()
: logger_(rclcpp::get_logger("scout_piper_scene_repr.semantic_collision"))
{
  if (!rclcpp::ok()) {
    RCLCPP_ERROR(logger_, "No ROS context: the semantic collision field cannot be received.");
    return;
  }
  node_ = std::make_shared<rclcpp::Node>("semantic_collision");
  logger_ = node_->get_logger();
  auto & c = config_;
  c.field_topic = node_->declare_parameter("field_topic", c.field_topic);
  c.require_field = node_->declare_parameter("require_field", c.require_field);
  c.unknown_is_occupied = node_->declare_parameter("unknown_is_occupied", c.unknown_is_occupied);
  c.check_soft = node_->declare_parameter("check_soft", c.check_soft);
  c.max_field_age_s = node_->declare_parameter("max_field_age_s", c.max_field_age_s);
  c.max_voxel_age_s = node_->declare_parameter("max_voxel_age_s", c.max_voxel_age_s);
  c.sphere_padding_m = node_->declare_parameter("sphere_padding_m", c.sphere_padding_m);
  c.continuous_step = node_->declare_parameter("continuous_step", c.continuous_step);
  c.ignore_links = node_->declare_parameter("ignore_links", std::vector<std::string>{});
  c.model_frame = node_->declare_parameter("model_frame", c.model_frame);
  model_frame_ = c.model_frame;

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, node_, false);
  sub_ = node_->create_subscription<FieldMsg>(
    c.field_topic, rclcpp::QoS(1).reliable().transient_local(),
    [this](FieldMsg::ConstSharedPtr msg) {onField(std::move(msg));});
  tf_timer_ = node_->create_wall_timer(
    std::chrono::milliseconds(100), [this]() {updateTransform();});

  executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  executor_->add_node(node_);
  spin_thread_ = std::thread([this]() {executor_->spin();});
  RCLCPP_INFO(
    logger_, "Listening for the semantic distance field on %s (require_field=%s, "
    "unknown_is_occupied=%s, max_voxel_age_s=%.1f).", c.field_topic.c_str(),
    c.require_field ? "true" : "false", c.unknown_is_occupied ? "true" : "false",
    c.max_voxel_age_s);
}

SemanticFieldListener::~SemanticFieldListener()
{
  if (executor_) {
    executor_->cancel();
  }
  if (spin_thread_.joinable()) {
    spin_thread_.join();
  }
}

void SemanticFieldListener::registerModelFrame(const std::string & frame)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (model_frame_.empty()) {
    model_frame_ = frame;
  }
}

void SemanticFieldListener::onField(FieldMsg::ConstSharedPtr msg)
{
  auto f = std::make_shared<SemanticDistanceField>();
  f->origin = {msg->origin.x, msg->origin.y, msg->origin.z};
  f->voxel_size = msg->voxel_size;
  f->size = {msg->size[0], msg->size[1], msg->size[2]};
  f->stamp = rclcpp::Time(msg->header.stamp, node_->get_clock()->get_clock_type()).seconds();
  f->frame_id = msg->header.frame_id;
  f->hard_classes = msg->hard_classes;
  f->hard_distance = msg->hard_distance;
  f->hard_class = msg->hard_class;
  f->soft_class = msg->soft_class;
  f->soft_max_penetration = msg->soft_max_penetration;
  f->soft_distance = msg->soft_distance;
  f->age_ds = msg->age_ds;
  const std::string err = f->finalize();
  if (!err.empty()) {
    RCLCPP_ERROR(logger_, "Dropping malformed semantic distance field: %s", err.c_str());
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool frame_changed = !field_ || field_->frame_id != f->frame_id;
    field_ = std::move(f);
    if (frame_changed) {
      transform_ok_ = false;
    }
  }
  updateTransform();
}

void SemanticFieldListener::updateTransform()
{
  std::string field_frame, model_frame;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!field_ || model_frame_.empty()) {
      return;
    }
    field_frame = field_->frame_id;
    model_frame = model_frame_;
  }
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  bool ok = true;
  if (field_frame != model_frame) {
    try {
      const auto ts = tf_buffer_->lookupTransform(field_frame, model_frame, tf2::TimePointZero);
      const auto & q = ts.transform.rotation;
      const auto & t = ts.transform.translation;
      T.linear() = Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized().toRotationMatrix();
      T.translation() = Eigen::Vector3d(t.x, t.y, t.z);
    } catch (const tf2::TransformException & e) {
      ok = false;
      RCLCPP_WARN_THROTTLE(
        logger_, *node_->get_clock(), 5000, "No TF %s <- %s for the semantic field: %s",
        field_frame.c_str(), model_frame.c_str(), e.what());
    }
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (field_ && field_->frame_id == field_frame) {
    // keep the last good transform on a lookup miss; only a new frame invalidates it
    if (ok) {
      field_T_model_ = T;
      transform_ok_ = true;
    }
  }
}

FieldSnapshot SemanticFieldListener::snapshot() const
{
  FieldSnapshot s;
  if (!node_) {
    s.reason = "no ROS context";
    return s;
  }
  s.now = node_->get_clock()->now().seconds();
  std::lock_guard<std::mutex> lock(mutex_);
  s.field = field_;
  s.field_T_model = field_T_model_;
  s.transform_ok = transform_ok_;
  if (!s.field) {
    s.reason = "no field received on " + config_.field_topic;
  } else if (s.now - s.field->stamp > config_.max_field_age_s) {
    s.reason = "field is " + std::to_string(s.now - s.field->stamp) + " s old";
  } else if (!s.transform_ok) {
    s.reason = "no transform " + s.field->frame_id + " <- " + model_frame_;
  }
  return s;
}

}  // namespace scout_piper_scene_repr
