# Quantization LLMs
 Quantization in machine learning is a technique used to reduce the computational and memory requirements of models by converting high-precision numerical values (like floating-point numbers) into lower-precision representations (such as integers). This process can significantly enhance the efficiency of deploying models, especially on resource-constrained devices like smartphones and embedded systems.Typically, models are trained using 32-bit floating-point numbers (FP32). Quantization reduces these to 16-bit floating-point (FP16) or 8-bit integers (INT8), among other formats.

 This repository contains a custom W8A16 Quantizer, that performs quantization by replacing the troch.nn.Linear layers in LLMs.
