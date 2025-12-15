**Set up Your Platform**

-   Python 3.10 (**MUST**)
-   Download and unzip OV and OV GenAI test build (**MUST**)
-   Download, unzip and install NPU Driver test build (**MUST**)

**Run sample**

Open Command-Prompt and run below commands

```bash
<OV_DIR>/setupvars.bat
<OV_GENAI_DIR>/setupvars.bat
python text2image_export_import_ws.py <model_name> <prompt>
```

-   Log:

    ```bash
    [ INFO ] text encoder is running on CPU.
    [ INFO ] unet is running on NPU.
    [ INFO ] vae decoder is running on CPU.
    [WARNING] 23:09:26.906 [NPUZeroInitStructsHolder] Some features might not be available! Plugin L0 API minor version = 13, Driver L0 API minor version = 14
    [WARNING] 23:09:48.686 [multi-cluster-strategy-assignment]     VPUNN error code ERROR_INPUT_TOO_BIG is caught, code val 4294967295
    [WARNING] 23:10:14.390 [calculate-async-region-cycle-cost]   Create new cost model instance
    [WARNING] 23:10:14.526 [print-nn-cache-statistics] Create new cost model instance
    [WARNING] 23:10:14.744 [feasible-allocation]   Create new cost model instance
    [WARNING] 23:10:15.753 [print-nn-cache-statistics] Create new cost model instance
    [WARNING] 23:10:17.522 [detect-dma-split-candidate]   Create new cost model instance
    [WARNING] 23:10:17.609 [print-nn-cache-statistics] Create new cost model instance
    [WARNING] 23:10:18.827 [simplify-schedule]   Create new cost model instance
    [WARNING] 23:10:19.041 [print-nn-cache-statistics] Create new cost model instance
    [WARNING] 23:10:19.102 [add-sw-kernel-instruction-prefetch]   Create new cost model instance
    [WARNING] 23:10:20.865 [inference-execution-analysis]   Create new cost model instance
    [WARNING] 23:10:20.935 [inference-execution-analysis]   [Energy] The following SHAVE operations are unsupported by VPUNN, which can influence the estimation: [builtin_Dequantize]
    [WARNING] 23:10:20.990 [print-nn-cache-statistics] Create new cost model instance
    [WARNING] 23:10:29.784 [calculate-async-region-cycle-cost]   Create new cost model instance
    [WARNING] 23:10:29.890 [print-nn-cache-statistics] Create new cost model instance
    [WARNING] 23:10:30.094 [feasible-allocation]   Create new cost model instance
    [WARNING] 23:10:31.837 [print-nn-cache-statistics] Create new cost model instance
    [WARNING] 23:10:33.531 [detect-dma-split-candidate]   Create new cost model instance
    [WARNING] 23:10:33.620 [print-nn-cache-statistics] Create new cost model instance
    [WARNING] 23:10:34.923 [simplify-schedule]   Create new cost model instance
    [WARNING] 23:10:35.130 [print-nn-cache-statistics] Create new cost model instance
    [WARNING] 23:10:35.197 [add-sw-kernel-instruction-prefetch]   Create new cost model instance
    [WARNING] 23:10:37.142 [inference-execution-analysis]   Create new cost model instance
    [WARNING] 23:10:37.200 [inference-execution-analysis]   [Energy] The following SHAVE operations are unsupported by VPUNN, which can influence the estimation: [builtin_Dequantize, cache_prefetch]
    [WARNING] 23:10:37.276 [print-nn-cache-statistics] Create new cost model instance
    [WARNING] 23:10:41.101 [calculate-async-region-cycle-cost]   Create new cost model instance
    [WARNING] 23:10:41.138 [print-nn-cache-statistics] Create new cost model instance
    [WARNING] 23:10:41.209 [feasible-allocation]   Create new cost model instance
    [WARNING] 23:10:41.617 [print-nn-cache-statistics] Create new cost model instance
    [WARNING] 23:10:42.206 [detect-dma-split-candidate]   Create new cost model instance
    [WARNING] 23:10:42.237 [print-nn-cache-statistics] Create new cost model instance
    [WARNING] 23:10:42.716 [simplify-schedule]   Create new cost model instance
    [WARNING] 23:10:42.799 [print-nn-cache-statistics] Create new cost model instance
    [WARNING] 23:10:42.827 [add-sw-kernel-instruction-prefetch]   Create new cost model instance
    [WARNING] 23:10:43.564 [inference-execution-analysis]   Create new cost model instance
    [WARNING] 23:10:43.579 [inference-execution-analysis]   [Energy] The following SHAVE operations are unsupported by VPUNN, which can influence the estimation: [builtin_Dequantize, cache_prefetch]
    [WARNING] 23:10:43.610 [print-nn-cache-statistics] Create new cost model instance
    [WARNING] 23:11:18.727 [calculate-async-region-cycle-cost]   Create new cost model instance
    [WARNING] 23:11:19.896 [print-nn-cache-statistics] Create new cost model instance
    [WARNING] 23:11:22.119 [feasible-allocation]   Create new cost model instance
    [WARNING] 23:11:42.476 [print-nn-cache-statistics] Create new cost model instance
    [WARNING] 23:12:03.748 [detect-dma-split-candidate]   Create new cost model instance
    [WARNING] 23:12:06.362 [print-nn-cache-statistics] Create new cost model instance
    [WARNING] 23:12:43.718 [simplify-schedule]   Create new cost model instance
    [WARNING] 23:12:48.416 [print-nn-cache-statistics] Create new cost model instance
    [WARNING] 23:12:50.067 [add-sw-kernel-instruction-prefetch]   Create new cost model instance
    [WARNING] 23:13:36.732 [inference-execution-analysis]   Create new cost model instance
    [WARNING] 23:13:37.622 [inference-execution-analysis]   There are 30 tasks with invalid cost, estimation might not be valid
    [WARNING] 23:13:37.796 [inference-execution-analysis]   [Energy] The following SHAVE operations are unsupported by VPUNN, which can influence the estimation: [builtin_Convert, builtin_MVN1MeanVar, builtin_MVN1Normalize, builtin_MVN1SumOp, builtin_ReduceSum, cache_prefetch]
    [WARNING] 23:13:39.475 [print-nn-cache-statistics] Create new cost model instance
    [ INFO ] Load model time: 318.2834529876709.
    [ INFO ] Export model time: 0.7023859024047852.
    [WARNING] 23:14:29.173 [NPUPlugin] The usage of a compiled model can lead to undefined behavior. Please use OpenVINO IR instead!
    [ INFO ] Load blob time: 3.282236337661743.
    [ INFO ] Inference time: 8.318743228912354.
    [WARNING] 23:14:42.162 [NPUZeroInitStructsHolder] zeContextDestroy failed to destroy the context; Level zero context was already destroyed
    ```


