# PC's GPU

> NVIDIA TU106M [GeForce RTX 2060 Mobile]

```bash
$ inxi -Gxxx -a
Graphics:
  Device-1: NVIDIA TU106M [GeForce RTX 2060 Mobile] vendor: ASUSTeK
    driver: nvidia v: 535.274.02 alternate: nvidiafb,nouveau,nvidia_drm
    non-free: 550.xx+ status: current (as of 2024-04; EOL~2026-12-xx)
    arch: Turing code: TUxxx process: TSMC 12nm FF built: 2018-2022 pcie:
    gen: 3 speed: 8 GT/s lanes: 8 link-max: lanes: 16 ports: active: none
    empty: DP-1 bus-ID: 01:00.0 chip-ID: 10de:1f15 class-ID: 0300
  Device-2: AMD Renoir [Radeon RX Vega 6 ] vendor: ASUSTeK driver: amdgpu
    v: kernel arch: GCN-5 code: Vega process: GF 14nm built: 2017-20 pcie:
    gen: 4 speed: 16 GT/s lanes: 16 ports: active: eDP-1 empty: HDMI-A-1
    bus-ID: 05:00.0 chip-ID: 1002:1636 class-ID: 0300 temp: 68.0 C
  Device-3: IMC Networks USB2.0 HD UVC WebCam driver: uvcvideo type: USB
    rev: 2.0 speed: 480 Mb/s lanes: 1 mode: 2.0 bus-ID: 3-4:3
    chip-ID: 13d3:56a2 class-ID: 0e02 serial: 0x0001
  Display: x11 server: X.Org v: 21.1.11 with: Xwayland v: 23.2.6
    compositor: gnome-shell v: 46.0 driver: X: loaded: amdgpu,nvidia
    unloaded: fbdev,modesetting,nouveau,radeon,vesa dri: radeonsi gpu: amdgpu
    display-ID: :1 screens: 1
  Screen-1: 0 s-res: 1920x1080 s-dpi: 102 s-size: 480x270mm (18.90x10.63")
    s-diag: 551mm (21.68")
  Monitor-1: eDP-1 mapped: eDP-1-0 model: ChiMei InnoLux 0x1521 built: 2020
    res: 1920x1080 hz: 144 dpi: 142 gamma: 1.2 size: 344x193mm (13.54x7.6")
    diag: 394mm (15.5") ratio: 16:9 modes: max: 1920x1080 min: 640x480
  API: EGL v: 1.5 hw: drv: nvidia drv: amd radeonsi platforms: device: 0
    drv: nvidia device: 1 drv: radeonsi device: 3 drv: swrast gbm:
    drv: kms_swrast surfaceless: drv: nvidia x11: drv: nvidia
    inactive: wayland,device-2
  API: OpenGL v: 4.6.0 compat-v: 4.5 vendor: nvidia mesa v: 535.274.02
    glx-v: 1.4 direct-render: yes renderer: NVIDIA GeForce RTX 2060/PCIe/SSE2
    memory: 5.86 GiB
```

# Markdown Exercise

This same document was formatted using markdown, it's source code can be found
in this repository
[https://github.com/sebascert/cuda-homework-1](https://github.com/sebascert/cuda-homework-1).
The file containing the markdown source is:

> [https://github.com/sebascert/cuda-homework-1/tree/main/src/body.md](https://github.com/sebascert/cuda-homework-1/tree/main/src/body.md).

# Local Running Sample

## Dependencies (requirements.txt)

```requirements
Mako==1.3.10
MarkupSafe==3.0.3
numpy==2.3.4
platformdirs==4.5.0
pycuda==2025.1.2
pytools==2025.2.5
siphash24==1.8
typing_extensions==4.15.0
```

## Code

```python
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
```

## Result

```bash
$ py cuda.py
Hola Mundo desde el hilo 0
Hola Mundo desde el hilo 1
Hola Mundo desde el hilo 2
Hola Mundo desde el hilo 3
Hola Mundo desde el hilo 4
Hola Mundo desde el hilo 5
Hola Mundo desde el hilo 6
Hola Mundo desde el hilo 7
Hola Mundo desde el hilo 8
Hola Mundo desde el hilo 9
Hola Mundo desde el hilo 10
Hola Mundo desde el hilo 11
```
