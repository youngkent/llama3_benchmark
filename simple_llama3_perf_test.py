# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import time
from typing import List

import fire

from llama import Llama


def run_once(
    generator,
    prompts,
    max_gen_len,
    temperature,
    top_p,
    print_output,
):
    st = time.time()
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    if print_output:
        for prompt, result in zip(prompts, results):
            print(prompt)
            print(f"> {result['generation']}")
            print("\n==================================\n")
    return time.time() - st


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_gen_len: int = 64,
    max_batch_size: int = 32,
    print_output: bool = False,
    test_iterations: int = 10,
    enable_torch_compile: bool = False,
):

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        enable_torch_compile=enable_torch_compile,
    )

    prompts: List[str] = ["I believe the meaning of life is"] * max_batch_size

    print(f"Warming up the model ...")
    for _ in range(3):
        run_once(generator, prompts, max_gen_len, temperature, top_p, print_output)

    print(f"Measuring perf ...")
    latencies = []
    throughputs = []
    for i in range(test_iterations):
        latency = run_once(
            generator,
            prompts,
            max_gen_len,
            temperature,
            top_p,
            print_output,
        )
        latencies.append(latency)
        throughput = len(prompts) / latency
        throughputs.append(throughput)

        print(
            f"Batch completed with total latency: {latency:.3f}s, QPS: {throughput:.3f}"
        )

    print(
        f"Average latency: {sum(latencies) / len(latencies):.3f}s, average QPS: {sum(throughputs) / len(throughputs):.3f}"
    )


if __name__ == "__main__":
    fire.Fire(main)
