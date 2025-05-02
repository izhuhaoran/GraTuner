tune_delta = False
hybrid_schedule = False
kernel_fusion_enable = True
num_vertices = 10000


def write_cfg_to_schedule(cfg):
    #write into a schedule file the configuration

    direction_0 = cfg['direction_0']
    if tune_delta:
        delta_0 = cfg['delta']
    dedup_0 = cfg['dedup_0']
    frontier_output_0 = cfg['frontier_output_0']
    pull_rep_0 = cfg['pull_rep_0']
    LB_0 = cfg['LB_0']

    new_schedule = "schedule:\n"

    new_schedule += "SimpleGPUSchedule s1;\n";
    if LB_0 == "EDGE_ONLY" and cfg['EB_0'] == "ENABLED":
        new_schedule += "s1.configLoadBalance(EDGE_ONLY, BLOCKED, " + str(int(int(num_vertices)/cfg['BS_0'])) + ");\n"
        direction_0 = "PUSH"
    else:
        new_schedule += "s1.configLoadBalance(" + LB_0 + ");\n"
    new_schedule += "s1.configFrontierCreation(" + frontier_output_0 + ");\n"
    if direction_0 == "PULL":
        new_schedule += "s1.configDirection(PULL, " + pull_rep_0 + ");\n"
    else:
        new_schedule += "s1.configDirection(PUSH);\n"
    if tune_delta:
        new_schedule += "s1.configDelta(" + str(delta_0) + ");\n"
    new_schedule += "s1.configDeduplication(" + dedup_0 + ");\n"

    if hybrid_schedule:
        direction_1 = cfg['direction_1']
        if tune_delta:
            delta_1 = cfg['delta']
        dedup_1 = cfg['dedup_1']
        frontier_output_1 = cfg['frontier_output_1']
        pull_rep_1 = cfg['pull_rep_1']
        LB_1 = cfg['LB_1']

        #threshold = hybrid_threshold
        threshold = cfg['threshold']
        
        new_schedule += "SimpleGPUSchedule s2;\n";
        new_schedule += "s2.configLoadBalance(" + LB_1 + ");\n"
        new_schedule += "s2.configFrontierCreation(" + frontier_output_1 + ");\n"
        if direction_1 == "PULL":
            new_schedule += "s2.configDirection(PULL, " + pull_rep_1 + ");\n"
        else:
            new_schedule += "s2.configDirection(PUSH);\n"
        if tune_delta:
            new_schedule += "s2.configDelta(" + str(delta_1) + ");\n"
        new_schedule += "s2.configDeduplication(" + dedup_1 + ");\n"
        
        new_schedule += "HybridGPUSchedule h1(INPUT_VERTEXSET_SIZE, " + str(threshold/1000) + ", s1, s2);\n"
        new_schedule += "program->applyGPUSchedule(\"s0:s1\", h1);\n"

    else:
        new_schedule += "program->applyGPUSchedule(\"s0:s1\", s1);\n"


    if kernel_fusion_enable:
        kernel_fusion = cfg['kernel_fusion']
        new_schedule += "SimpleGPUSchedule s0;\n"
        new_schedule += "s0.configKernelFusion(" + kernel_fusion + ");\n"
        new_schedule += "program->applyGPUSchedule(\"s0\", s0);\n"

    print (cfg)
    #print (new_schedule)

    new_schedule_file_name = 'schedule_2' 
    #print (new_schedule_file_name)
    f1 = open (new_schedule_file_name, 'w')
    f1.write(new_schedule)
    f1.close()
    
cfg = {'kernel_fusion': 'DISABLED', 'LB_0': 'WM', 'EB_0': 'None', 'BS_0': 0, 'direction_0': 'PULL', 'dedup_0': 'ENABLED', 'frontier_output_0': 'UNFUSED_BOOLMAP', 'pull_rep_0': 'BOOLMAP'}

write_cfg_to_schedule(cfg)