#!/bin/bash
# graph_list = ['sx-stackoverflow', 'dblp-cite', 'dbpedia-team', 'dimacs9-E', 'douban',
#               'facebook-wosn-wall', 'github', 'komarix-imdb', 'moreno_blogs', 'opsahl-usairport',
#               'patentcite', 'petster-friendships-dog', 'roadNet-CA', 'subelj_cora', 'sx-mathoverflow',
#                'youtube-groupmemberships', 
#               ]
# graph_list = ['youtube-u-growth', 'dblp_coauthor', 'soc-LiveJournal1', 'orkut-links', 'roadNet-CA', ]

# algo_list = ['sssp', 'cc', 'pagerank', 'bfs']

# 生成所有图，所有算法的自动调优结果
# for graph in sx-stackoverflow dblp-cite github patentcite youtube-groupmemberships sx-mathoverflow roadNet-CA subelj_cora facebook-wosn-wall dbpedia-team dimacs9-E douban komarix-imdb moreno_blogs opsahl-usairport petster-friendships-dog 
# for graph in youtube-u-growth dblp_coauthor soc-LiveJournal1 orkut-links roadNet-CA
# do
#     for algo in pagerank bfs sssp cc
#     do
#         /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/start_auto_YiTian.sh ${algo} ${graph} 6 > /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/${algo}_${graph}_6.log
#         /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/start_auto_YiTian.sh ${algo} ${graph} 7 > /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/${algo}_${graph}_7.log
#         /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/start_auto_YiTian.sh ${algo} ${graph} 8 > /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/${algo}_${graph}_8.log
#         /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/start_auto_YiTian.sh ${algo} ${graph} 9 > /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/${algo}_${graph}_9.log
#         /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/start_auto_YiTian.sh ${algo} ${graph} 0 > /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/${algo}_${graph}_0.log
#     done
# done

# for graph in youtube-u-growth dblp_coauthor soc-LiveJournal1 orkut-links roadNet-CA
# do
#     for algo in pagerank bfs sssp cc
#     do
#         for round in $(seq 0 9)
#         do
#             /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/start_auto_YiTian.sh ${algo} ${graph} ${round} > /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/${algo}_${graph}_${round}.log
#         done
#     done
# done

for round in $(seq 10 19)
do
    for graph in youtube-u-growth dblp_coauthor soc-LiveJournal1 orkut-links roadNet-CA
    do
        for algo in pagerank bfs sssp cc
        do
            /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/start_auto_YiTian.sh ${algo} ${graph} ${round} > /home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/${algo}_${graph}_${round}.log
        done
    done
done



