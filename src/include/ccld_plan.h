#ifndef NCCL_CCLD_PLAN_H_
#define NCCL_CCLD_PLAN_H_

#include <cstdint>

enum class CcldCollective : uint8_t {
  AllReduce,
  AllGather,
  ReduceScatter,
  Other
};

enum class CcldAlgorithm : uint8_t {
  Ring,
  Tree,
  Other
};

enum class CcldWorkType : uint8_t {
  Collective,
  CollectiveRegistered,
  Other
};

enum class CcldPlanReason : uint8_t {
  Supported,
  Persistent,
  MultipleCollectives,
  P2pPresent,
  UnsupportedCollective,
  UnsupportedAlgorithm,
  UnexpectedWorkType,
  MultipleWork,
  ChannelRange,
  TreeGroups
};

struct CcldPlanDescriptor {
  bool persistent = false;
  bool multipleCollectives = false;
  bool p2pPresent = false;
  bool multipleWork = false;
  CcldCollective collective = CcldCollective::Other;
  CcldAlgorithm algorithm = CcldAlgorithm::Other;
  CcldWorkType workType = CcldWorkType::Other;
  uint32_t channelLo = 0;
  uint32_t channelHi = 0;
  uint32_t frameChannels = 0;
  uint32_t frameGroups = 0;
};

struct CcldPlanDecision {
  CcldPlanReason reason;
  bool supported() const { return reason == CcldPlanReason::Supported; }
};

inline CcldPlanDecision ccldPlanDecision(const CcldPlanDescriptor& plan) {
  if (plan.persistent) return {CcldPlanReason::Persistent};
  if (plan.multipleCollectives) return {CcldPlanReason::MultipleCollectives};
  if (plan.p2pPresent) return {CcldPlanReason::P2pPresent};
  if (plan.collective != CcldCollective::AllReduce &&
      plan.collective != CcldCollective::AllGather &&
      plan.collective != CcldCollective::ReduceScatter) {
    return {CcldPlanReason::UnsupportedCollective};
  }
  if (plan.algorithm != CcldAlgorithm::Ring &&
      !(plan.collective == CcldCollective::AllReduce && plan.algorithm == CcldAlgorithm::Tree)) {
    return {CcldPlanReason::UnsupportedAlgorithm};
  }
  if (plan.workType != CcldWorkType::Collective &&
      plan.workType != CcldWorkType::CollectiveRegistered) {
    return {CcldPlanReason::UnexpectedWorkType};
  }
  if (plan.multipleWork) return {CcldPlanReason::MultipleWork};
  if (plan.frameChannels == 0 || plan.channelLo > plan.channelHi ||
      plan.channelHi >= plan.frameChannels) {
    return {CcldPlanReason::ChannelRange};
  }
  if (plan.algorithm == CcldAlgorithm::Tree && plan.frameGroups < 2) {
    return {CcldPlanReason::TreeGroups};
  }
  return {CcldPlanReason::Supported};
}

inline const char* ccldPlanReasonString(CcldPlanReason reason) {
  switch (reason) {
    case CcldPlanReason::Supported: return "supported";
    case CcldPlanReason::Persistent: return "persistent";
    case CcldPlanReason::MultipleCollectives: return "multiple_collectives";
    case CcldPlanReason::P2pPresent: return "p2p_present";
    case CcldPlanReason::UnsupportedCollective: return "unsupported_collective";
    case CcldPlanReason::UnsupportedAlgorithm: return "unsupported_algorithm";
    case CcldPlanReason::UnexpectedWorkType: return "unexpected_work_type";
    case CcldPlanReason::MultipleWork: return "multiple_work";
    case CcldPlanReason::ChannelRange: return "channel_range";
    case CcldPlanReason::TreeGroups: return "tree_groups";
  }
  return "unknown";
}

#endif
