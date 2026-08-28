# Attentive Eraser Windows CMD README Design

## Goal

Add a focused Windows Command Prompt guide for building and running the C++ Attentive Eraser sample with SD1.5 and SDXL models on CPU and GPU.

## Location

Create `samples/cpp/image_generation/README_attentive_eraser_windows.md` beside the sample source and existing image-generation README.

## Content

The guide will:

- use Windows CMD syntax rather than PowerShell or Bash;
- define one editable workspace-root environment variable to avoid repeating machine-specific absolute paths;
- document prerequisites, including CMake, Visual Studio C++ tools, OpenVINO 2026.4, and the two exported model directories;
- show how to remove and recreate `build_attn_eraser`, configure Release mode, and build only `attentive_eraser_pipeline` and its dependencies;
- add both the built OpenVINO GenAI DLL directory and the installed OpenVINO runtime DLL directory to `PATH`;
- show SD1.5 and SDXL commands for both CPU and GPU, using the existing Torch reference image and mask paths;
- run each case from a separate output directory and rename `object_removed_image.bmp` so results are not overwritten;
- explain the command-line arguments, output dimensions, default behavior, and expected CPU/GPU performance difference;
- include short troubleshooting entries for missing DLL error `0xC0000135`, missing `OpenVINOConfig.cmake`, and high SDXL CPU memory use.

## Scope

This change is documentation-only. It will not modify the sample, build system, model files, or generated results. It will not add automation scripts because the requested artifact is a readable CMD guide.

## Validation

Validate that every referenced repository path exists, every command uses valid CMD syntax, and the documented executable and arguments match `attentive_eraser_pipeline.cpp`. Review the final Markdown for broken relative links and placeholders.