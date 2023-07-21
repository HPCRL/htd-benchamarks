transposed_gemm_i, transposed_gemm_j, transposed_gemm_k = tuple(transposed_gemm.op.axis) + tuple(transposed_gemm.op.reduce_axis)
out_i, out_j = tuple(out.op.axis) + tuple(out.op.reduce_axis)
transposed_gemm_i_o_i, transposed_gemm_i_i = s[transposed_gemm].split(transposed_gemm_i, factor=8)
transposed_gemm_i_o_o_i, transposed_gemm_i_o_i = s[transposed_gemm].split(transposed_gemm_i_o_i, factor=2)
transposed_gemm_i_o_o_o_i, transposed_gemm_i_o_o_i = s[transposed_gemm].split(transposed_gemm_i_o_o_i, factor=2)
transposed_gemm_i_o_o_o_o, transposed_gemm_i_o_o_o_i = s[transposed_gemm].split(transposed_gemm_i_o_o_o_i, factor=1)
transposed_gemm_j_o_i, transposed_gemm_j_i = s[transposed_gemm].split(transposed_gemm_j, factor=1)
transposed_gemm_j_o_o_i, transposed_gemm_j_o_i = s[transposed_gemm].split(transposed_gemm_j_o_i, factor=1)
transposed_gemm_j_o_o_o_i, transposed_gemm_j_o_o_i = s[transposed_gemm].split(transposed_gemm_j_o_o_i, factor=16)
transposed_gemm_j_o_o_o_o, transposed_gemm_j_o_o_o_i = s[transposed_gemm].split(transposed_gemm_j_o_o_o_i, factor=1)
transposed_gemm_k_o_i, transposed_gemm_k_i = s[transposed_gemm].split(transposed_gemm_k, factor=8)
transposed_gemm_k_o_o, transposed_gemm_k_o_i = s[transposed_gemm].split(transposed_gemm_k_o_i, factor=1)
s[transposed_gemm].reorder(transposed_gemm_i_o_o_o_o, transposed_gemm_j_o_o_o_o, transposed_gemm_i_o_o_o_i, transposed_gemm_j_o_o_o_i, transposed_gemm_i_o_o_i, transposed_gemm_j_o_o_i, transposed_gemm_k_o_o, transposed_gemm_k_o_i, transposed_gemm_i_o_i, transposed_gemm_j_o_i, transposed_gemm_k_i, transposed_gemm_i_i, transposed_gemm_j_i)
out_i_o_i, out_i_i = s[out].split(out_i, factor=16)
out_i_o_o_i, out_i_o_i = s[out].split(out_i_o_i, factor=2)
out_i_o_o_o, out_i_o_o_i = s[out].split(out_i_o_o_i, factor=1)
out_j_o_i, out_j_i = s[out].split(out_j, factor=1)
out_j_o_o_i, out_j_o_i = s[out].split(out_j_o_i, factor=16)
out_j_o_o_o, out_j_o_o_i = s[out].split(out_j_o_o_i, factor=1)
s[out].reorder(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i, out_j_o_o_i, out_i_o_i, out_j_o_i, out_i_i, out_j_i)
s[transposed_gemm].compute_at(s[out], out_j_o_i)
B_shared = s.cache_read(B, "shared", [transposed_gemm])
B_shared_ax0, B_shared_ax1 = tuple(B_shared.op.axis)
s[B_shared].compute_at(s[transposed_gemm], transposed_gemm_k_o_o)
A_shared = s.cache_read(A, "shared", [transposed_gemm])
A_shared_ax0, A_shared_ax1 = tuple(A_shared.op.axis)
s[A_shared].compute_at(s[transposed_gemm], transposed_gemm_k_o_o)
out_i_o_o_o_j_o_o_o_fused = s[out].fuse(out_i_o_o_o, out_j_o_o_o)
s[out].bind(out_i_o_o_o_j_o_o_o_fused, te.thread_axis("blockIdx.x"))
out_i_o_o_i_j_o_o_i_fused = s[out].fuse(out_i_o_o_i, out_j_o_o_i)
s[out].bind(out_i_o_o_i_j_o_o_i_fused, te.thread_axis("vthread"))
out_i_o_i_j_o_i_fused = s[out].fuse(out_i_o_i, out_j_o_i)
s[out].bind(out_i_o_i_j_o_i_fused, te.thread_axis("threadIdx.x"))
B_shared_ax0_ax1_fused = s[B_shared].fuse(B_shared_ax0, B_shared_ax1)
B_shared_ax0_ax1_fused_o, B_shared_ax0_ax1_fused_i = s[B_shared].split(B_shared_ax0_ax1_fused, factor=2)
s[B_shared].vectorize(B_shared_ax0_ax1_fused_i)
B_shared_ax0_ax1_fused_o_o, B_shared_ax0_ax1_fused_o_i = s[B_shared].split(B_shared_ax0_ax1_fused_o, factor=32)
s[B_shared].bind(B_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))
A_shared_ax0_ax1_fused = s[A_shared].fuse(A_shared_ax0, A_shared_ax1)
A_shared_ax0_ax1_fused_o, A_shared_ax0_ax1_fused_i = s[A_shared].split(A_shared_ax0_ax1_fused, factor=4)
s[A_shared].vectorize(A_shared_ax0_ax1_fused_i)
A_shared_ax0_ax1_fused_o_o, A_shared_ax0_ax1_fused_o_i = s[A_shared].split(A_shared_ax0_ax1_fused_o, factor=32)
s[A_shared].bind(A_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))
s[transposed_gemm].pragma(transposed_gemm_i_o_o_o_o, "auto_unroll_max_step", 64)
s[transposed_gemm].pragma(transposed_gemm_i_o_o_o_o, "unroll_explicit", True)
