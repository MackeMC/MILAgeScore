import torch
print(torch.__version__)
print("Compute capability:", torch.cuda.get_device_capability(0))  # expect (12, 0)
x = torch.randn(1000, 1000, device='cuda')
y = torch.mm(x, x)
print("GPU compute OK:", y.shape)  # expect torch.Size([1000, 1000])
