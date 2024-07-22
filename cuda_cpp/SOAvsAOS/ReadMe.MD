# AOS vs SOA for GPU and CPU Performance Comparison

## Comparison Table

| Feature                    | Array of Structures (AOS)                                  | Structure of Arrays (SOA)                                |
|----------------------------|------------------------------------------------------------|----------------------------------------------------------|
| **Memory Layout**          | Elements are stored contiguously with all fields together. | Each field of the structure is stored in a separate array. |
| **Memory Coalescing (GPU)**| Less efficient due to scattered accesses for each field.   | More efficient due to contiguous accesses for each field. |
| **Ease of Use**            | Easier and more intuitive for developers.                  | More complex and less intuitive for developers.           |
| **SIMD Efficiency (GPU)**  | Lower SIMD efficiency due to non-coalesced accesses.       | Higher SIMD efficiency due to coalesced accesses.         |
| **Cache Efficiency (GPU)** | Lower cache efficiency due to scattered accesses.          | Higher cache efficiency due to contiguous accesses.       |
| **Alignment and Padding**  | Potential issues with alignment and padding.               | Avoids alignment and padding issues.                      |
| **Typical Use Case**       | Suitable for CPU and simple GPU applications.              | Suitable for performance-critical GPU applications.       |
| **Performance on GPU**     | Generally lower on GPUs.                                   | Generally higher on GPUs.                                 |
| **Performance on CPU**     | Generally higher on CPUs.                                  | Generally lower on CPUs due to scattered memory accesses. |
| **Access Pattern**         | Better for interleaved access to multiple fields.          | Better for access to individual fields in bulk.           |

## Example Usage

### AOS

```cpp
struct Vec3 {
    float x, y, z;
};

__global__ void vecAddAOS(Vec3* a, Vec3* b, Vec3* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i].x = a[i].x + b[i].x;
        c[i].y = a[i].y + b[i].y;
        c[i].z = a[i].z + b[i].z;
    }
}

void vecAddAOS_CPU(Vec3* a, Vec3* b, Vec3* c, int size) {
    for (int i = 0; i < size; ++i) {
        c[i].x = a[i].x + b[i].x;
        c[i].y = a[i].y + b[i].y;
        c[i].z = a[i].z + b[i].z;
    }
}
