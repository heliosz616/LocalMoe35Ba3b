# Provenance and licensing scope

The runtime/operator code was extracted from a local Qwen INT8 deployment
research implementation and reorganized without intentionally changing the
accepted operator arithmetic. FreeToken's expert offload, double buffering,
release/ready synchronization and routing design informed this implementation.
The inspected FreeToken reference revision was
`0ab982f10905fa775962a4eddcb44caa50065251`. This project is not FreeToken itself.

Transformers supplies the Qwen3_5Moe architecture and unmodified GDN arithmetic;
PyTorch, Triton, Accelerate, Safetensors and the CUDA runtime remain separate
dependencies with their own licenses. No third-party model weights, tokenizer
assets or datasets are redistributed here. Obtain permission and comply with
the original model/data licenses before using or redistributing your artifacts.

This publication does not assign a new open-source license to the project code.
Public availability should not be interpreted as an additional license grant.
The repository owner may choose a project license in a subsequent release.
