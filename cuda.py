import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

# Kernel de CUDA
mod = SourceModule("""
__global__ void hola_mundo_kernel()
{
int idx = threadIdx.x;
if (idx < 12)
printf("Hola Mundo desde el hilo %d\\n", idx);
}
""")
# Obtenemos la función del kernel
hola_mundo = mod.get_function("hola_mundo_kernel")
# Ejecutamos el kernel: 1 bloque con 12 hilos
hola_mundo(block=(12, 1, 1), grid=(1, 1))
