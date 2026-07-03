# refer to https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/setup.py
import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

try:
    from torch_npu.utils.cpp_extension import NpuExtension
except ImportError:
    NpuExtension = None

requirements = ["torch", "torchvision"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
    source_npu = glob.glob(os.path.join(extensions_dir, "npu", "*.cpp"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    if CUDA_HOME is not None and torch.cuda.is_available():
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    elif NpuExtension is not None and torch.npu.is_available():
        extension = NpuExtension
        sources += source_npu
        define_macros += [("WITH_NPU", None)]
        extra_compile_args["npu"] = [
            "-D__NPU_NO_HALF_OPERATORS__",
            "-D__NPU_NO_HALF_CONVERSIONS__",
            "-D__NPU_NO_HALF2_OPERATORS__",
        ]
    else:
        raise NotImplementedError("Cuda or NPU is not available")

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "MultiScaleDeformableAttention",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


setup(
    name="MultiScaleDeformableAttention",
    version="1.0",
    author="Hao Wang",
    url="https://github.com/wanghao9610/X-SAM",
    description="PyTorch Wrapper for CUDA and NPU Functions of Multi-Scale Deformable Attention",
    packages=find_packages(
        exclude=(
            "configs",
            "tests",
        )
    ),
    ext_modules=get_extensions(),
    cmdclass={
        # "build_ext": torch.utils.cpp_extension.BuildExtension.with_options(use_ninja=False) # disable ninja for npu build
        "build_ext": torch.utils.cpp_extension.BuildExtension
    },
)
