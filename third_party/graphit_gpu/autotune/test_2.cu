#include "gpu_intrinsics.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;
int32_t __delta_param;
gpu_runtime::GraphT<char> __device__ edges;
gpu_runtime::GraphT<char> __host_edges;
gpu_runtime::GraphT<char> __device__ edges__transposed;
gpu_runtime::GraphT<char> __host_edges__transposed;
int32_t __device__ *parent;
int32_t *__host_parent;
int32_t *__device_parent;
void __device__ parent_generated_vector_op_apply_func_0(int32_t v);
void __device__ updateEdge(int32_t src, int32_t dst, gpu_runtime::VertexFrontier __output_frontier);
bool toFilter(int32_t v);
void __device__ reset(int32_t v);
template <typename EdgeWeightType>
void __device__ gpu_operator_body_2(gpu_runtime::GraphT<EdgeWeightType> graph, int32_t src, int32_t dst, int32_t edge_id, gpu_runtime::VertexFrontier input_frontier, gpu_runtime::VertexFrontier output_frontier) {
	// Body of the actual operator
	if (!input_frontier.d_byte_map_input[dst])
		return;
	updateEdge(dst, src, output_frontier);
}
void __device__ parent_generated_vector_op_apply_func_0(int32_t v) {
	parent[v] = -(1);
}
void __device__ updateEdge(int32_t src, int32_t dst, gpu_runtime::VertexFrontier __output_frontier) {
	bool result_var1 = 0;
	result_var1 = gpu_runtime::CAS(&parent[dst], -(1), src);
	if (result_var1) {
		gpu_runtime::enqueueVertexBytemap(__output_frontier.d_byte_map_output, __output_frontier.d_num_elems_output, dst);
	}
}
bool __host__ toFilter(int32_t v) {
	bool output;
	cudaMemcpy(__host_parent + v, __device_parent + v, sizeof(int32_t), cudaMemcpyDeviceToHost);
	output = (__host_parent[v]) == (-(1));
	return output;
}
void __device__ reset(int32_t v) {
	parent[v] = -(1);
}
int __host__ main(int argc, char* argv[]) {
	__delta_param = 1;
	gpu_runtime::load_graph(__host_edges, argv[1], false);
	cudaMemcpyToSymbol(edges, &__host_edges, sizeof(__host_edges), 0, cudaMemcpyHostToDevice);
	__host_edges__transposed = gpu_runtime::builtin_transpose(__host_edges);
	cudaMemcpyToSymbol(edges__transposed, &__host_edges__transposed, sizeof(__host_edges__transposed), 0, cudaMemcpyHostToDevice);
	cudaMalloc(&__device_parent, gpu_runtime::builtin_getVertices(__host_edges) * sizeof(int32_t));
	cudaMemcpyToSymbol(parent, &__device_parent, sizeof(int32_t*), 0);
	__host_parent = new int32_t[gpu_runtime::builtin_getVertices(__host_edges)];
	gpu_runtime::vertex_set_apply_kernel<gpu_runtime::AccessorAll, parent_generated_vector_op_apply_func_0><<<NUM_CTA, CTA_SIZE>>>(__host_edges.getFullFrontier());
	for (int32_t trail = 0; trail < 10; trail++) {
		gpu_runtime::VertexFrontier frontier = gpu_runtime::create_new_vertex_set(gpu_runtime::builtin_getVertices(__host_edges), 0);
		startTimer();
		gpu_runtime::vertex_set_apply_kernel<gpu_runtime::AccessorAll, reset><<<NUM_CTA, CTA_SIZE>>>(__host_edges.getFullFrontier());
		int32_t start_vertex = atoi(argv[2]);
		gpu_runtime::builtin_addVertex(frontier, start_vertex);
		__host_parent[start_vertex] = start_vertex;
		cudaMemcpy(__device_parent + start_vertex, __host_parent + start_vertex, sizeof(int32_t), cudaMemcpyHostToDevice);
		while ((gpu_runtime::builtin_getVertexSetSize(frontier)) != (0)) {
			gpu_runtime::VertexFrontier output;
			{
				gpu_runtime::vertex_set_prepare_boolmap(frontier);
				output = frontier;
