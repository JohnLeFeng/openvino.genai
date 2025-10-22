import sys
import time
import logging
import argparse

import openvino as ov
import openvino_genai


logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout) 
log = logging.getLogger()


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    # fmt: off
    args.add_argument('-h', '--help', action = 'help',
                      help='Show this help message and exit.')
    args.add_argument('-m','--model_dir',type = str, default = "Llama-3.1-8B-Instruct-ov", required = False,
                      help='Optional. Specify the LLM model directory. Default is Llama-3.1-8B-Instruct-ov.')
    args.add_argument('-prefix', '--enable_prefix', default = True, action=argparse.BooleanOptionalAction,
                      help='Optional. Whether to enable prefix cache.')
    args.add_argument('-pl','--prompt_length',type = int, default = 1024, required = False,
                      help='Optional. Number of prompt length.')
    args.add_argument('-pl2','--prompt_length_2', required = False,
                      help='Optional. Number of prompt length of second round.')
    return parser.parse_args()


def streamer(subword):
    print(subword, end='', flush=True)
    # Return flag corresponds whether generation should be stopped.
    return openvino_genai.StreamingStatus.RUNNING

def main():
    args = parse_args()

    device = "GPU"
    model_dir = args.model_dir


    scheduler_config = openvino_genai.SchedulerConfig()
    scheduler_config.num_kv_blocks = 784
    # scheduler_config.max_num_batched_tokens = 256
    scheduler_config.max_num_batched_tokens = 131072000
    # scheduler_config.max_num_seqs = 256
    scheduler_config.max_num_seqs = 1
    scheduler_config.dynamic_split_fuse = False
    if (args.enable_prefix):
        log.info("Enabled prefix cache.")
        scheduler_config.enable_prefix_caching = True
    else:
        log.info("Disabled prefix cache.")
        scheduler_config.enable_prefix_caching = False

    generation_config = openvino_genai.GenerationConfig()
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 50

    ov_config = {"CACHE_DIR": "./cache"}
    log.info("Pipeline will be initialized with ContinuousBatchingPipeline")
    llm_pipe = openvino_genai.ContinuousBatchingPipeline(model_dir, scheduler_config, device, properties=ov_config)

    tokenizer = llm_pipe.get_tokenizer()

    text_prompt = "Intel (NASDAQ: INTC) Surges to $38 as Nvidia’s $5 Billion Stake and U.S. Support Fuel AI Foundry Comeback Ahead of Earnings\nIntel (NASDAQ: INTC) continues its remarkable rebound, closing at $38.02, up 2.72% on Monday, marking a fresh two-year high and an 85% year-to-date gain. The rally, which began in early August, has been fueled by a powerful combination of AI-driven optimism, strategic investments, and renewed confidence in Intel’s foundry business. Despite this surge, analysts remain divided over whether the comeback is sustainable or simply an overreaction to headlines ahead of the company’s Q3 earnings release on October 23.\nMorgan Stanley added fuel to the discussion this week by raising its price target to $36 from $23, while maintaining an Equal Weight rating on the stock. The firm cited improving sentiment around Intel’s foundry strategy but warned that the stock’s 100% rally since August may be overextended. According to Morgan Stanley’s note, Intel’s upcoming earnings could beat the “low consensus bar,” but long-term conviction still hinges on proving execution strength across its manufacturing and AI divisions.\nBehind this surge lies a string of headline-making partnerships that have reshaped investor perception. In September, Nvidia (NASDAQ: NVDA) took a $5 billion equity stake in Intel, acquiring roughly 4% of the company in what many view as a landmark cross-industry collaboration. The investment sparked a 22.8% single-day rally, Intel’s biggest percentage gain since 1987. The companies plan to co-develop chips combining Intel CPUs and Nvidia GPUs, creating hybrid designs aimed at the booming AI data center market. This alliance, once unimaginable between fierce rivals, highlights Nvidia’s confidence in Intel’s technological capabilities and the foundry’s long-term relevance.\nAdding to the momentum, the U.S. government converted $10 billion in CHIPS Act subsidies into equity, effectively taking a 9.9% ownership stake in Intel. Washington’s participation underscores the company’s importance to national semiconductor independence and strategic defense manufacturing. Mean while, SoftBank’s Vision Fund contributed an additional $2 billion, and rumors of talks with Apple over potential AI hardware partnerships sent shares up another 6% in late September, although no deal has been confirmed. Collectively, these moves have re-established Intel as a focal point of U.S. industrial policy and global chip supply-chain reform.\nStill, the euphoria contrasts sharply with Intel’s financial reality. For the second quarter of 2025, the company reported revenue of $12.86 billion, up just 0.2% year-over-year, with a net loss of $2.92 billion, an 81% decline from last year. Gross margins remain under 30%, a far cry from the 55% level once considered Intel’s hallmark of efficiency. On a brighter note, earnings per share improved to –$0.10 from –$0.63, the best sequential recovery since 2022, and EBITDA rose 63% to $2.52 billion, showing tangible progress in cost management.\nCEO Lip-Bu Tan has responded with decisive restructuring efforts. Intel is cutting approximately 20% of its workforce, reducing headcount to 75,000 employees by year-end, while temporarily halting construction on select fabs to preserve cash. The company’s cash and short-term investments fell 27.6% to $21.21 billion, signaling the financial strain of its global expansion projects. Yet Tan insists that these measures are necessary to support the 18A process technology, Intel’s most advanced manufacturing node to date, expected to rival TSMC’s 2-nanometer process by 2026."

    prompt_length = args.prompt_length
    if args.prompt_length_2:
        prompt_length_2 = int(args.prompt_length_2)
    else:
        log.info("2nd prompt will reuse 1st prompt.")
        prompt_length_2 = prompt_length

    tokens = tokenizer.encode([text_prompt])
    log.info("Total token length: %d.", tokens.input_ids.shape[1])

    input_data = [ov.Tensor([tokens.input_ids.data[0, :prompt_length]])]
    input_data_2 = [ov.Tensor([tokens.input_ids.data[0, :prompt_length_2]])]

    log.info("Processing 1st round inference")
    t_start = time.time()
    first_output = llm_pipe.generate(input_data, [generation_config])
    log.info("Process time of 1st round: %.3f s", time.time() - t_start)

    log.info("Processing 2nd round inference")
    t_start = time.time()
    second_output = llm_pipe.generate(input_data_2, [generation_config])
    log.info("Process time of 2nd round: %.3f s", time.time() - t_start)

    first_generated_token = []
    second_generated_token = []
    for res in first_output:
        first_generated_token.append(res.m_generation_ids[0])

    for res in second_output:
        second_generated_token.append(res.m_generation_ids[0])

    if prompt_length == prompt_length_2:
        if first_generated_token != second_generated_token:
            log.info("1st round and 2nd round result is different!")
            log.info("1st output:")
            log.info("  %s", tokenizer.decode(first_generated_token))
            log.info("2nd output:")
            log.info("  %s", tokenizer.decode(second_generated_token))
        else:
            log.info("1st round and 2nd round result is same.")

if '__main__' == __name__:
    main()
