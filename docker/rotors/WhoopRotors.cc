// A thrust stand approximation for ducted 31 mm props. The upstream
// Betaflight plugin drives the shafts; measured shaft speed determines thrust.
#include <array>
#include <cmath>
#include <string>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/JointVelocity.hh>

namespace sim = gz::sim;
using gz::math::Vector3d;
class WhoopRotors final : public sim::System, public sim::ISystemConfigure,
                          public sim::ISystemPreUpdate {
  struct Rotor {
    std::string name;
    Vector3d arm;
    sim::Entity shaft = sim::kNullEntity;
  };
  std::array<Rotor, 4> rotors{{{"rear_right", {-.022981,-.022981,0}},
                               {"front_right",{ .022981,-.022981,0}},
                               {"rear_left",  {-.022981, .022981,0}},
                               {"front_left", { .022981, .022981,0}}}};
  sim::Link body;
  double thrustCoefficient = 3e-7;
  double torqueArm = .003;
public:
  void Configure(const sim::Entity &entity, const std::shared_ptr<const sdf::Element> &params,
                 sim::EntityComponentManager &ecm, sim::EventManager &) override {
    sim::Model model(entity);
    body = sim::Link(model.LinkByName(ecm, "body"));
    body.EnableVelocityChecks(ecm);
    thrustCoefficient = params->Get<double>("thrust_coefficient", thrustCoefficient).first;
    torqueArm = params->Get<double>("torque_arm", torqueArm).first;
    for (auto &r : rotors)
      r.shaft = model.JointByName(ecm, r.name + "_shaft");
  }

  void PreUpdate(const sim::UpdateInfo &info, sim::EntityComponentManager &ecm) override {
    if (info.paused) return;
    Vector3d force, torque;
    for (const auto &r : rotors) {
      auto vel = ecm.Component<sim::components::JointVelocity>(r.shaft);
      if (!vel || vel->Data().empty()) continue;
      double omega = vel->Data()[0];
      Vector3d lift(0, 0, thrustCoefficient * omega * omega);
      force += lift;
      torque += r.arm.Cross(lift);
      torque.Z() -= std::copysign(torqueArm * lift.Z(), omega);
    }
    auto orientation = sim::worldPose(body.Entity(), ecm).Rot();
    // Blade flapping and duct drag oppose body rotation. Scale damping
    // with rotor speed (sqrt of thrust), so a stopped prop adds no torque.
    if (auto omega = body.WorldAngularVelocity(ecm)) {
      auto local = orientation.RotateVectorReverse(*omega);
      double diskSpeed = std::sqrt(force.Z() / .255);
      torque -= diskSpeed * Vector3d(.00010 * local.X(), .00010 * local.Y(),
                                     .000055 * local.Z());
    }
    Vector3d drag;
    if (auto velocity = body.WorldLinearVelocity(ecm))
      drag = -.0015 * velocity->Length() * (*velocity);
    body.AddWorldWrench(ecm, orientation.RotateVector(force) + drag,
                        orientation.RotateVector(torque));
  }
};
GZ_ADD_PLUGIN(WhoopRotors, sim::System, sim::ISystemConfigure, sim::ISystemPreUpdate)
