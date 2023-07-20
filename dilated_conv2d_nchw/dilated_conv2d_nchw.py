import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python


@auto_scheduler.register_workload
def dilated_conv2d(N, H, W, CO, CI, KH, KW, stride, padding, dilation):
   # Create placeholders for the input and filter tensors
    input_tensor = te.placeholder((N, CI, H + 2 * padding, W + 2 * padding), name='input')
    filter_tensor = te.placeholder((CO, CI, KH, KW), name='filter')

    # Create an output tensor
    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1
    output_tensor = te.placeholder((N, CO, OH, OW), name='output')

    # Define the computation of the Conv2D operation with accumulation, stride, and padding
    output = topi.nn.conv2d_nchw(input_tensor, filter_tensor, stride=stride, padding='VALID', dilation=dilation)
    accumulated_output = te.compute(
       (N, CO, OH, OW),
    lambda n, co, oh, ow: output(n, co, oh, ow) + output_tensor(n, co, oh, ow),
    name='accumulated_output'
    ) 
    #data = te.placeholder((N, CI, H + 2 * padding, W + 2 * padding), name="data")
    #kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    #conv = topi.nn.dilated_conv2d_nchw(data, kernel, stride, padding='VALID', dilation=1, out_dtype="float32")
    return [input_tensor, filter_tensor, output_tensor, accumulated_output]

def tune_problem(task):
    global log_file
    print(log_file)
    tune_number = int(sys.argv[12])
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=tune_number,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    task.tune(tune_option)

def test_result(sch, args, target):
    func = tvm.build(sch, args, target)
    OH = (H + 2 * padding - KH) // strides + 1
    OW = (W + 2 * padding - KW) // strides + 1
    data_np = np.random.uniform(size=(N, CI, H + 2 * padding, W + 2 * padding)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    output_np = np.random.uniform(size=(N, CO, OH, OW)).astype(np.float32)
    conv_np = conv2d_nchw_python(data_np, weight_np, strides, padding='VALID', dilation)
    out_np = conv_np + output_np

    dev = tvm.cuda()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    output_tvm = tvm.nd.array(output_np, device=dev)
    out_tvm = tvm.nd.empty(out_np.shape, device=dev)
    func(data_tvm, weight_tvm, output_tvm, out_tvm)

    # Check results
    np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, output_tvm, out_tvm).results) * 1000)
    )
    return "Execution time of this operator: %.3f ms" % (np.median(evaluator(data_tvm, weight_tvm, output_tvm, out_tvm).results) * 1000)
 

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
H = int(sys.argv[3])
W = int(sys.argv[4])
CO = int(sys.argv[5])
CI = int(sys.argv[6])
KH = int(sys.argv[7])
KW = int(sys.argv[8])
strides = int(sys.argv[9])
padding = int(sys.argv[10])
dilation = int(sys.argv[11])

target = tvm.target.Target("cuda")
print(f"arc={target.arch}")

task = tvm.auto_scheduler.SearchTask( func=dilated_conv2d, args=(N, H, W, CO, CI, KH, KW, strides, padding, dilation), target=target)

print("Computational DAG:")
print(task.compute_dag)

log_file = f"evaluate/dilated_conv2d_nchw_N{N}_H{H}_W{W}_CO{CO}_CI{CI}_KH{KH}_KW{KW}_strides_{strides}_padding_{padding}_dilation_{dilation}_{target.arch}.json"
cuda_code = f"evaluate/dilated_conv2d_nchw_N{N}_H{H}_W{W}_CO{CO}_CI{CI}_KH{KH}_KW{KW}_strides_{strides}_padding_{padding}_dilation_{dilation}_{target.arch}.cu"
schedule_code = f"evaluate/dilated_conv2d_nchw_N{N}_H{H}_W{W}_CO{CO}_CI{CI}_KH{KH}_KW{KW}_strides_{strides}_padding_{padding}_dilation_{dilation}_{target.arch}.py"
eval_res = f"evaluate/dilated_conv2d_nchw_N{N}_H{H}_W{W}_CO{CO}_CI{CI}_KH{KH}_KW{KW}_strides_{strides}_padding_{padding}_dilation_{dilation}_{target.arch}.txt"

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
