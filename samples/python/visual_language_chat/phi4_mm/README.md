# Enable Phi4 multimodel with OpenVINO™

-----

1. Convert model to IR with Intel Optimum

    ```sh
    optimum-cli export openvino --model microsoft/Phi-4-multimodal-instruct --trust-remote-code --weight-format fp16 --task image-text-to-text Phi-4-multimodal-instruct-ov
    ```

2. Compress language model

    * Apply group compression

        * INT4_SYM

            ```sh
            python compress_lm.py -m Phi-4-multimodal-instruct-ov -gs 64
            ```

    * Apply channel-wise quantization

        * INT4_SYM

            ```sh
            python compress_lm.py -m Phi-4-multimodal-instruct-ov -gs -1
            ```
        
        * NF4

            ```sh
            python compress_lm.py -m Phi-4-multimodal-instruct-ov -gs -1 -p NF4
            ```
