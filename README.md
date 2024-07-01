# Quantization LLMs
## Quantization
Quantization in machine learning is a technique used to reduce the computational and memory requirements of models by converting high-precision numerical values (like floating-point numbers) into lower-precision representations (such as integers). This process can significantly enhance the efficiency of deploying models, especially on resource-constrained devices like smartphones and embedded systems.Typically, models are trained using 32-bit floating-point numbers (FP32). Quantization reduces these to 16-bit floating-point (FP16) or 8-bit integers (INT8), among other formats.

## Quantization Repository
This repository contains a custom W8A16 Quantizer designed to perform quantization by replacing the torch.nn.Linear layers in Large Language Models (LLMs). The repository includes four Jupyter notebooks on:
- Quantization Basics: Demonstrates the fundamental workings of quantization.
- Quantization Granularities: Explores different levels of granularity in quantization.
- Custom Quantizers: Defines custom quantizers tailored for specific needs.
- Quantizing LLMs: Applies the custom quantizers to quantize Large Language Models effectively.

Additionally, the repository includes helper.py, which contains all the different functions defined across the four notebooks to facilitate understanding and reuse of the quantization methods.

___
Feel free to explore the notebooks and helper functions for detailed implementations and insights into the quantization process.
