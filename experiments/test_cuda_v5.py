import torch
import torch.nn as nn

print("1. PyTorch imported", flush=True)
print("PyTorch:", torch.__version__, flush=True)
print("CUDA runtime:", torch.version.cuda, flush=True)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("2. GPU:", torch.cuda.get_device_name(0), flush=True)

print("3. Creating 3D model on GPU", flush=True)

model = nn.Sequential(
    nn.Conv3d(1, 8, kernel_size=3, padding=1),
    nn.GroupNorm(8, 8),
    nn.LeakyReLU(0.1),
    nn.Conv3d(8, 1, kernel_size=1),
).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
scaler = torch.amp.GradScaler("cuda")

print("4. Creating test input", flush=True)

x = torch.randn(
    1, 1, 64, 64, 64,
    device="cuda"
)

target = torch.rand(
    1, 1, 64, 64, 64,
    device="cuda"
)

print("5. Running AMP forward pass", flush=True)

optimizer.zero_grad(set_to_none=True)

with torch.autocast("cuda", dtype=torch.float16):
    prediction = model(x)
    loss = nn.functional.mse_loss(prediction.float(), target)

print("6. Running backward pass", flush=True)

scaler.scale(loss).backward()

print("7. Updating model", flush=True)

scaler.step(optimizer)
scaler.update()

torch.cuda.synchronize()

print("8. CUDA TEST PASSED", flush=True)
print("Loss:", loss.item(), flush=True)
print("GPU memory used:", torch.cuda.memory_allocated() / 1024**2, "MB", flush=True)