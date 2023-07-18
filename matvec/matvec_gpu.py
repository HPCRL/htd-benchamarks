import numpy as np
import tvm
from tvm import te, auto_scheduler


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matvec_add(N, L, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L,), name="B", dtype=dtype)
    C = te.placeholder((N,), name="C", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    matvec = te.compute(
        (N,),
        lambda i: te.sum(A[i, k] * B[k], axis=k),
        name="matvec"
    )
    out = te.compute((N,), lambda i: matvec[i] + C[i], name="out")

    return [A, B, C, out]

def tune_problem(task):
    tune_number = int(sys.argv[4])
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=tune_number,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=0,
    )
    task.tune(tune_option)

def test_result(sch, args, target):
    func = tvm.build(sch, args, target)
    a_np = np.random.uniform(size=(N, L)).astype(np.float32)
    b_np = np.random.uniform(size=(L,)).astype(np.float32)
    c_np = np.random.uniform(size=(N,)).astype(np.float32)
    out_np = a_np.dot(b_np) + c_np

    dev = tvm.cuda()
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.array(c_np, device=dev)
    out_tvm = tvm.nd.empty(out_np.shape, device=dev)
    func(a_tvm, b_tvm, c_tvm, out_tvm)

    # Check results
    np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(a_tvm, b_tvm, c_tvm, out_tvm).results) * 1000)
    )
    return "Execution time of this operator: %.3f ms" % (np.median(evaluator(a_tvm, b_tvm, c_tvm, out_tvm).results) * 1000)
 

def resume_search(task, log_file):
    print("Resume search:")
    cost_model = auto_scheduler.XGBModel()
    cost_model.update_from_file(log_file)
    search_policy = auto_scheduler.SketchPolicy(
        task, cost_model, init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
    )
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=5, measure_callbacks=[auto_scheduler.RecordToFile(log_file)]
    )
    task.tune(tune_option, search_policy=search_policy)



import os
import sys
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"

exec_mode = sys.argv[1]
N = int(sys.argv[2])
L = int(sys.argv[3])

target = tvm.target.Target("cuda")
print(f"arc={target.arch}")

task = tvm.auto_scheduler.SearchTask(func=matvec_add, args=(N, L, "float32"), target=target)

print("Computational DAG:")
print(task.compute_dag)

log_file = f"../../evaluate/matvec/matvec_M{N}_K{L}_{target.arch}.json"
cuda_code = f"../../evaluate/matvec/matvec_M{N}_K{L}_{target.arch}.cu"
schedule_code = f"../../evaluate/matvec/matvec_M{N}_K{L}_{target.arch}.py"
eval_res = f"../../evaluate/matvec/matvec_M{N}_K{L}_{target.arch}.txt"

if exec_mode == "tune":
    tune_problem(task)

sch, args = task.apply_best(log_file)
if exec_mode == "resume":
    resume_search(task, log_file)

elif exec_mode == "eval":
    test_result(sch, args, target)

elif exec_mode == "timing":
    test_result(sch, args, target)

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))

print("Equivalent python schedule:")
print(task.print_best(log_file))

print("Equivalent CUDA code:")
print(task.print_best(log_file, "cuda"))

if exec_mode != "eval":
    with open(schedule_code, 'w') as f:
        f.write(task.print_best(log_file))
    with open(cuda_code, 'w') as f:
        f.write(task.print_best(log_file, "cuda"))
