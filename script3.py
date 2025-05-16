import torch
# set a fixed seed for this usage of pytorch
torch.manual_seed(42)

from pyblaz.compression import PyBlaz
import time

#torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cudnn.allow_tf32 = True

# Create a compressor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
codec = PyBlaz(
    block_shape=(128, 128),
    dtype=torch.float32,
    device=device,
    compute_mode="tf32",
    compile=True
)


#->>F Create a tensor with extreme values
print("--- Creating a tensor with extreme TF32 max values and compressing/decompressing it ---")
#tf32_max = (2 - 2 ** -10) * (1 << 127)

tf32_big = (2 - 2 ** -10) * (1 << 64)

initRndTens = torch.randn(2048, 2048, device=device)

x = initRndTens * tf32_big

# x = torch.ones(2048, 2048, device=device) * tf32_max

start_time = time.time()

#with torch.cuda.amp.autocast():
#    compressed_x = codec.compress(x)
#    decompressed_x = codec.decompress(compressed_x)

compressed_x = codec.compress(x)
decompressed_x = codec.decompress(compressed_x)

print(f"dec output partial =", decompressed_x[0:16])

# does x have a nan or inf?
if (x.isnan().sum() > 0):
    print("x has a nan!")
if (x.isinf().sum() > 0):
    print("x has an inf!")    
    

# does decompressed_x have a nan or inf?
    
end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.6f} seconds")
