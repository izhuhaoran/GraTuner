from itertools import product


tune_delta_enable = False
kernel_fusion_enable = True
edge_only_enable = True
hybrid_schedule_enble = False


def generate_config_combinations(config_dict:dict):
    # 获取所有的键名
    keys = config_dict.keys()
    # 生成所有的值组合
    value_combinations = product(*config_dict.values())
    # 为每个值组合生成一个字典
    dict_combinations = [dict(zip(keys, values)) for values in value_combinations]
    print (f'before unique config lens: {len(dict_combinations)}')
    # print(dict_combinations[0])
    for combination in dict_combinations:
        # 应用条件性逻辑
        if combination['LB_0'] != 'EDGE_ONLY':
            combination['EB_0'] = 'None'
            combination['BS_0'] = 0
            
        if combination['LB_0'] == 'EDGE_ONLY':
            combination['direction_0'] = 'PUSH'
            
        if combination['EB_0'] != 'ENABLED':
            combination['BS_0'] = 0
                
        if combination['direction_0'] != 'PULL':
            combination['pull_rep_0'] = 'None'
        
        if hybrid_schedule_enble:
            if combination['direction_1'] != 'PULL':
                combination['pull_rep_1'] = 'None'
    
    # 去重
    unique_dict_combinations = [dict(t) for t in {tuple(d.items()) for d in dict_combinations}]
    
    return unique_dict_combinations

def generate_config_combinations_v2(config_dict:dict):
    # 获取所有的键名
    keys = config_dict.keys()
    # 生成所有的值组合
    value_combinations = product(*config_dict.values())
    # 为每个值组合生成一个字典
    dict_combinations = [dict(zip(keys, values)) for values in value_combinations]
    print (f'before unique config lens: {len(dict_combinations)}')
    # print(dict_combinations[0])
    for combination in dict_combinations:
        # 应用条件性逻辑
        if combination['LB_0'] != 'EDGE_ONLY':
            combination['EB_0'] = 'None'
            combination['BS_0'] = 0
            
        if combination['LB_0'] == 'EDGE_ONLY' and combination['EB_0'] == 'ENABLED':
            combination['direction_0'] = 'PUSH'
            
        if combination['EB_0'] != 'ENABLED':
            combination['BS_0'] = 0
                
        if combination['direction_0'] != 'PULL':
            combination['pull_rep_0'] = 'None'
        
        if hybrid_schedule_enble:
            if combination['direction_1'] != 'PULL':
                combination['pull_rep_1'] = 'None'
    
    # 去重
    unique_dict_combinations = [dict(t) for t in {tuple(d.items()) for d in dict_combinations}]
    
    return unique_dict_combinations


def test_all():
    config_dict : dict[str, list] = {}  # 所有可选配置集合

    if kernel_fusion_enable:
        config_dict['kernel_fusion'] = ['DISABLED', 'ENABLED']

    if edge_only_enable:
        config_dict['LB_0'] = ['VERTEX_BASED','TWC', 'TWCE', 'WM', 'CM', 'EDGE_ONLY']
        config_dict['EB_0'] = ['ENABLED', 'DISABLED']
        config_dict['BS_0'] = list(range(1, 21))
        # config_dict['BS_0'] = [5, 10, 15, 20]
    else:
        config_dict['LB_0'] = ['VERTEX_BASED','TWC', 'TWCE', 'WM', 'CM']
        
    config_dict['direction_0'] = ['PUSH', 'PULL']
    config_dict['dedup_0'] = ['ENABLED', 'DISABLED']
    config_dict['frontier_output_0'] = ['FUSED', 'UNFUSED_BITMAP', 'UNFUSED_BOOLMAP']
    config_dict['pull_rep_0'] = ['BITMAP', 'BOOLMAP']

    if hybrid_schedule_enble:
        config_dict['LB_1'] = ['VERTEX_BASED','TWC', 'TWCE', 'WM', 'CM']
        
        config_dict['direction_1'] = ['PUSH', 'PULL']
        config_dict['dedup_1'] = ['ENABLED', 'DISABLED']
        config_dict['frontier_output_1'] = ['FUSED', 'UNFUSED_BITMAP', 'UNFUSED_BOOLMAP']
        config_dict['pull_rep_1'] = ['BITMAP', 'BOOLMAP']
        
        # # We also choose the hybrid schedule threshold here
        # config_dict['threshold'] = list(range(0, 1001))
        # config_dict['threshold'] = [10, 50, 100, 200, 500, 1000, 1500, 2000]
        # config_dict['threshold'] = [10, 50, 100, 200, 500, 1000]
        config_dict['threshold'] = [500, 1000]


    
    # 生成所有配置组合的cfg字典
    config_combinations = generate_config_combinations(config_dict)
    config_combinations_v2 = generate_config_combinations_v2(config_dict)
        
    config_dict_v3 = config_dict
    config_dict_v3['LB_0'] = ['EDGE_ONLY']
    config_dict_v3['direction_0'] = ['PULL']
    config_dict_v3['EB_0'] = ['DISABLED']
    
    config_combinations_v3 = generate_config_combinations_v2(config_dict_v3)
    
    
    print (f'unique config lens: {len(config_combinations)}')
    print (f'unique config lens v2: {len(config_combinations_v2)}')
    
    print (f'unique config lens v3: {len(config_combinations_v3)}')
    
    def dict_lists_equal(list1, list2, list3):
        # Convert list of dicts to list of sorted tuples
        def dicts_to_sorted_tuples(lst):
            return sorted([tuple(sorted(d.items())) for d in lst])

        # Compare the converted lists
        return dicts_to_sorted_tuples(list1) == dicts_to_sorted_tuples(list2 + list3)


    if(dict_lists_equal(config_combinations_v2, config_combinations, config_combinations_v3)):
        print('pass')

test_all()