#include <graphit/frontend/high_level_schedule.h>
namespace graphit {
void user_defined_schedule (graphit::fir::high_level_schedule::ProgramScheduleNode::Ptr program) {
    program->configApplyDirection("s1", "SparsePush-DensePull");
    program->configApplyParallelization("s1", "dynamic-vertex-parallel");
    program->configApplyNumSSG("s1", "fixed-vertex-count", 17, "DensePull");
    program->configApplyPriorityUpdateDelta("s1", 354984 );
    program->configApplyPriorityUpdate("s1", "lazy_priority_update" );
    program->configApplyDenseVertexSet("s1","bitvector", "src-vertexset", "DensePull");}
}