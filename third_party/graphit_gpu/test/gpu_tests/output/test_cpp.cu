#include "gpu_intrinsics.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;
int32_t __delta_param;
gpu_runtime::GraphT<int32_t> __device__ edges;
gpu_runtime::GraphT<int32_t> __host_edges;
int32_t __device__ *SP;
int32_t *__host_SP;
int32_t *__device_SP;
void __device__ SP_generated_vector_op_apply_func_0(int32_t v);
void __device__ updateEdge(int32_t src, int32_t dst, int32_t weight, gpu_runtime::VertexFrontier __output_frontier);
void __device__ reset(int32_t v);
template <typename EdgeWeightType>
void __device__ gpu_operator_body_2(gpu_runtime::GraphT<EdgeWeightType> graph, int32_t src, int32_t dst, int32_t edge_id, gpu_runtime::VertexFrontier input_frontier, gpu_runtime::VertexFrontier output_frontier) {
	// Body of the actual operator code
	EdgeWeightType weight = graph.d_edge_weight[edge_id];
	updateEdge(src, dst, weight, output_frontier);
}
void __device__ SP_generated_vector_op_apply_func_0(int32_t v) {
	SP[v] = 2147483647;
}
void __device__ updateEdge(int32_t src, int32_t dst, int32_t weight, gpu_runtime::VertexFrontier __output_frontier) {
	bool result_var1 = 0;
	result_var1 = gpu_runtime::writeMin(&SP[dst], (SP[src] + weight));
	if (result_var1) {
		gpu_runtime::enqueueVertexSparseQueue(__output_frontier.d_sparse_queue_output, __output_frontier.d_num_elems_output, dst);
	}
}
void __device__ reset(int32_t v) {
	SP[v] = 2147483647;
}
int __host__ main(int argc, char* argv[]) {
	__delta_param = 1;
	gpu_runtime::load_graph(__host_edges, argv[1], false);
	cudaMemcpyToSymbol(edges, &__host_edges, sizeof(__host_edges), 0, cudaMemcpyHostToDevice);
	cudaMalloc(&__device_SP, gpu_runtime::builtin_getVertices(__host_edges) * sizeof(int32_t));
	cudaMemcpyToSymbol(SP, &__device_SP, sizeof(int32_t*), 0);
	__host_SP = new int32_t[gpu_runtime::builtin_getVertices(__host_edges)];
	gpu_runtime::vertex_set_apply_kernel<gpu_runtime::AccessorAll, SP_generated_vector_op_apply_func_0><<<NUM_CTA, CTA_SIZE>>>(__host_edges.getFullFrontier());
	gpu_runtime::vertex_set_apply_kernel<gpu_runtime::AccessorAll, reset><<<NUM_CTA, CTA_SIZE>>>(__host_edges.getFullFrontier());
	int32_t n = gpu_runtime::builtin_getVertices(__host_edges);
	gpu_runtime::VertexFrontier frontier = gpu_runtime::create_new_vertex_set(gpu_runtime::builtin_getVertices(__host_edges), 0);
	int32_t start_vertex = atoi(argv[2]);
	gpu_runtime::builtin_addVertex(frontier, start_vertex);
	__host_SP[start_vertex] = 0;
	cudaMemcpy(__device_SP + start_vertex, __host_SP + start_vertex, sizeof(int32_t), cudaMemcpyHostToDevice);
	int32_t rounds = 0;
	while ((gpu_runtime::builtin_getVertexSetSize(frontier)) != (0)) {
		gpu_runtime::VertexFrontier output;
		{
			gpu_runtime::vertex_set_prepare_sparse(frontier);
			output = frontier;
			gpu_runtime::vertex_based_load_balance_host<int32_t, gpu_operator_body_2, gpu_runtime::AccessorSparse, gpu_runtime::true_function>(__host_edges, frontier, output);
			cudaDeviceSynchronize();
			gpu_runtime::swap_queues(output);
			output.format_ready = gpu_runtime::VertexFrontier::SPARSE;
		}
		gpu_runtime::dedup_frontier_perfect(output);
		frontier = output;
		rounds = (rounds + 1);
		if ((rounds) == (n)) {
			break;
		}
	}
	gpu_runtime::deleteObject(frontier);
	for (int32_t vid = 0; vid < n; vid++) {
		cudaMemcpy(__host_SP + vid, __device_SP + vid, sizeof(int32_t), cudaMemcpyDeviceToHost);
		std::cout << __host_SP[vid] << std::endl;
	}
}

