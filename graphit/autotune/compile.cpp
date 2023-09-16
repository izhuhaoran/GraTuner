#include <graphit/frontend/high_level_schedule.h>
namespace graphit {
void user_defined_schedule (graphit::fir::high_level_schedule::ProgramScheduleNode::Ptr program) {
    program->configApplyDirection("s1", "SparsePush");
    program->configApplyParallelization("s1", "static-vertex-parallel");}
}