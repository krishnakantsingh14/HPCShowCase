# Dynamic parallelism


# How to Compile CUDA Code with Dynamic Parallelism

This guide provides instructions for compiling CUDA code that uses dynamic parallelism. Dynamic parallelism allows a CUDA kernel to launch other kernels from the device (GPU) side.

## Prerequisites

- CUDA Toolkit installed
- NVIDIA GPU that supports dynamic parallelism (Compute Capability 3.5 or higher)
- NVCC compiler

## Compilation Instructions

### Command Line (Bash)

If you are compiling from the command line using Bash, use the following command:

```bash
nvcc -o my_program my_program.cu -rdc=true
```


## Visual Studio

If you are using Visual Studio, follow these steps:

1. Right-click on your project in **Solution Explorer** and select **Properties**.
2. Go to **Configuration Properties -> CUDA C/C++ -> Common**.
3. Set **Generate Relocatable Device Code** to **Yes (-rdc=true)**.
4. Apply the changes and rebuild your project.

# Troubleshooting

- **Error: "kernel launch from `__device__` or `__global__` functions requires separate compilation mode"**  
  This error occurs when the `-rdc=true` flag is not set. Ensure that you have enabled relocatable device code as described above.

- **Device Does Not Support Dynamic Parallelism**  
  Make sure your GPU has a Compute Capability of 3.5 or higher. You can check your GPU's capabilities by using the following code in your program:

  ```cpp
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
  ```