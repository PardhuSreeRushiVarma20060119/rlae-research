# GPU Safe Shutdown & Cleanup Guide

## 1. During Script Execution
- The provided scripts use `torch.cuda.empty_cache()` implicitly via process termination, which is the safest way to clear VRAM.
- If running interactively (Jupyter), run this after every experiment block:
  ```python
  import torch
  import gc
  model = None
  tokenizer = None
  gc.collect()
  torch.cuda.empty_cache()
  ```

## 2. After Experiments (Cloud)
**CRITICAL**: Cloud providers charge by the minute/hour.
1. **Download Data**:
   - Compres your logs: `tar -czvf results.tar.gz experiments/logs/`
   - Download `results.tar.gz` to your local machine.
2. **Verify Download**: Open the archive locally to ensure files are valid.
3. **Terminate Instance**:
   - **Colab**: Runtime -> Disconnect and Delete Runtime.
   - **RunPod**: Go to Pods dashboard -> Click Stop -> Click Terminate (Trash icon). *Stopping* still charges for storage, *Terminating* stops all charges.
   - **Lambda/AWS**: Terminate the specific instance ID.

## 3. Emergency Cleanup
If a script hangs or GPU memory is "stuck":
1. Open terminal.
2. Run `nvidia-smi` to find the Process ID (PID).
3. Run `kill -9 <PID>` to force kill the process.
