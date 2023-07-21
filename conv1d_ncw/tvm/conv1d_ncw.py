import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv1d_ncw_python

@auto_scheduler.register_workload
def conv1d(N, W, CO, CI, KW, stride, padding):
    # Create placeholders for the input and filter tensors
    input_tensor = te.placeholder((N, CI, W + 2 * padding), name='input')
    filter_tensor = te.placeholder((CO, CI, KW), name='filter')

    # Create an output tensor
    OW = (W + 2 * padding - KW) // stride + 1
    output_tensor = te.placeholder((N, CO, OW), name='output')

    # Define the computation of the Conv1D operation with accumulation, stride, and padding
    output = topi.nn.conv1d_ncw(input_tensor, filter_tensor, stride=stride, padding='VALID', dilation=1)
    accumulated_output = te.compute(
        (N, CO, OW),
    lambda n, co, ow: output(n, co, ow) + output_tensor(n, co, ow),
    name='accumulated_output'
    )

    return [input_tensor, filter_tensor, output_tensor, accumulated_output]


def tune_problem(task):
    tune_number = int(sys.argv[9])
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=tune_number,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=0,
    )
    task.tune(tune_option)

def test_result(sch, args, target):
    func = tvm.build(sch, args, target)
    data_np = np.random.uniform(size=(N, CI, W + 2 * padding)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KW)).astype(np.float32)
    conv_np = conv1d_ncw_python(data_np, weight_np, strides, padding='VALID')
    out_np = np.maximum(conv_np, 0.0)

    dev = tvm.cuda()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(out_np.shape, device=dev)
    func(data_tvm, weight_tvm, out_tvm)

    # Check results
    np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )
    return "Execution time of this operator: %.3f ms" % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
 

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
W = int(sys.argv[3])
CO = int(sys.argv[4])
CI = int(sys.argv[5])
KW = int(sys.argv[6])
strides = int(sys.argv[7])
padding = int(sys.argv[8])

target = tvm.target.Target("cuda")
print(f"arc={target.arch}")

task = tvm.auto_scheduler.SearchTask( func=conv2d, args=(N, W, CO, CI, KW, strides, padding), target=target)

print("Computational DAG:")
print(task.compute_dag)

log_file = f"evaluate/conv1d_ncw_N{N}_W{W}_CO{CO}_CI{CI}_KW{KW}_strides_{strides}_padding_{padding}_{target.arch}.json"
cuda_code = f"evaluate/conv1d_ncw_N{N}_W{W}_CO{CO}_CI{CI}_KW{KW}_strides_{strides}_padding_{padding}_{target.arch}.cu"
schedule_code = f"evaluate/conv1d_ncw_N{N}_W{W}_CO{CO}_CI{CI}_KW{KW}_strides_{strides}_padding_{padding}_{target.arch}.py"
eval_res = f"evaluate/conv1d_ncw_N{N}_W{W}_CO{CO}_CI{CI}_KW{KW}_strides_{strides}_padding_{padding}_{target.arch}.txt"

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
