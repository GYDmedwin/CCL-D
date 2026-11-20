#include <assert.h>
#include <string>

#include "ccld_plan.h"

static CcldPlanDescriptor supported(CcldCollective collective, CcldAlgorithm algorithm,
                                    uint32_t groups = 1) {
  CcldPlanDescriptor plan;
  plan.collective = collective;
  plan.algorithm = algorithm;
  plan.workType = CcldWorkType::Collective;
  plan.channelLo = 1;
  plan.channelHi = 3;
  plan.frameChannels = 8;
  plan.frameGroups = groups;
  return plan;
}

static void test_supported_matrix() {
  assert(ccldPlanDecision(supported(CcldCollective::AllReduce, CcldAlgorithm::Ring)).supported());
  assert(ccldPlanDecision(supported(CcldCollective::AllGather, CcldAlgorithm::Ring)).supported());
  assert(ccldPlanDecision(supported(CcldCollective::ReduceScatter, CcldAlgorithm::Ring)).supported());
  assert(ccldPlanDecision(supported(CcldCollective::AllReduce, CcldAlgorithm::Tree, 2)).supported());

  CcldPlanDescriptor registered = supported(CcldCollective::AllReduce, CcldAlgorithm::Ring);
  registered.workType = CcldWorkType::CollectiveRegistered;
  assert(ccldPlanDecision(registered).supported());
}

static void test_rejections() {
  CcldPlanDescriptor plan = supported(CcldCollective::AllReduce, CcldAlgorithm::Ring);
  plan.persistent = true;
  assert(ccldPlanDecision(plan).reason == CcldPlanReason::Persistent);
  plan = supported(CcldCollective::AllReduce, CcldAlgorithm::Ring);
  plan.multipleCollectives = true;
  assert(ccldPlanDecision(plan).reason == CcldPlanReason::MultipleCollectives);
  plan = supported(CcldCollective::AllReduce, CcldAlgorithm::Ring);
  plan.p2pPresent = true;
  assert(ccldPlanDecision(plan).reason == CcldPlanReason::P2pPresent);
  plan = supported(CcldCollective::Other, CcldAlgorithm::Ring);
  assert(ccldPlanDecision(plan).reason == CcldPlanReason::UnsupportedCollective);
  plan = supported(CcldCollective::AllGather, CcldAlgorithm::Tree, 2);
  assert(ccldPlanDecision(plan).reason == CcldPlanReason::UnsupportedAlgorithm);
  plan = supported(CcldCollective::AllReduce, CcldAlgorithm::Other);
  assert(ccldPlanDecision(plan).reason == CcldPlanReason::UnsupportedAlgorithm);
  plan = supported(CcldCollective::AllReduce, CcldAlgorithm::Ring);
  plan.workType = CcldWorkType::Other;
  assert(ccldPlanDecision(plan).reason == CcldPlanReason::UnexpectedWorkType);
  plan = supported(CcldCollective::AllReduce, CcldAlgorithm::Ring);
  plan.multipleWork = true;
  assert(ccldPlanDecision(plan).reason == CcldPlanReason::MultipleWork);
  plan = supported(CcldCollective::AllReduce, CcldAlgorithm::Ring);
  plan.channelHi = 8;
  assert(ccldPlanDecision(plan).reason == CcldPlanReason::ChannelRange);
  plan = supported(CcldCollective::AllReduce, CcldAlgorithm::Tree, 1);
  assert(ccldPlanDecision(plan).reason == CcldPlanReason::TreeGroups);
}

static void test_reason_strings() {
  assert(ccldPlanReasonString(CcldPlanReason::Supported) == std::string("supported"));
  assert(ccldPlanReasonString(CcldPlanReason::P2pPresent) == std::string("p2p_present"));
  assert(ccldPlanReasonString(CcldPlanReason::TreeGroups) == std::string("tree_groups"));
}

int main() {
  test_supported_matrix();
  test_rejections();
  test_reason_strings();
  return 0;
}
