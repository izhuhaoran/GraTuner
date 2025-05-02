#include <graphit/frontend/high_level_schedule.h>
namespace graphit {
using namespace graphit::fir::gpu_schedule;
void user_defined_schedule (graphit::fir::high_level_schedule::ProgramScheduleNode::Ptr program) {
SimpleGPUSchedule s1;
s1.configLoadBalance(TWC);
s1.configFrontierCreation(FUSED);
s1.configDirection(PULL, BITMAP);
s1.configDeduplication(ENABLED);
program->applyGPUSchedule("s0:s1", s1);
SimpleGPUSchedule s0;
s0.configKernelFusion(DISABLED);
program->applyGPUSchedule("s0", s0);
}
}