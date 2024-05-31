import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

# Define the model and inputs
model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

# Profile the model with multiple iterations
with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
    on_trace_ready=tensorboard_trace_handler('./log/resnet18_multi_iter')
) as prof:
    for i in range(10):  # Profile 10 iterations
        model(inputs)
        prof.step()  # Mark the end of an iteration

# Print profiling results sorted by CPU memory usage
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
# Export the profiling results to a JSON file for further analysis
prof.export_chrome_trace("trace_multi_iter.json")


from torch.profiler import profile, tensorboard_trace_handler

with profile(..., on_trace_ready=tensorboard_trace_handler("./log/resnet18")) as prof:

    ...