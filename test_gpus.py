import torch

print("\n--- Vast.ai Multi-GPU Sanity Check ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"Total GPUs Detected: {gpu_count}\n")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} | VRAM: {round(props.total_memory / 1e9, 2)} GB")
    if gpu_count == 4:
        print("\nSUCCESS: All 4 GPUs are ready for Qwen2-VL Subagents!")
else:
    print("\nERROR: CUDA is not available. You are running on CPU!")
