# GPT2-rs

A Rust project for running inference (text generation) using GPT-2.

The purpose is to gain a deeper understanding of LLMs and transformer models in general. Therefore, everything is implemented from scratch, without using any higher-level machine-learning or math libraries. (With the exception of the BPE tokenizer, which seemed less interesting to implement.)

The code runs on the CPU and is not optimized, except that we run matrix multiplication multi-threaded on all available cores. The goal here is not performance, but understanding. The small variant of GPT-2 is used, which gives reasonable performance on a decent CPU. Compiler optimizations make a big difference, so the `--release` flag is necessary for good performance.

## Prerequisites

- Rust
- Cargo (comes with Rust)

## Download the Model

The model file is provided as a release, due to its size (475 MB). [Download](https://github.com/jojju/llm-rs/releases/download/Model_file/gpt2_small_124M.bin) it and put it at the top level of the repository.

## Usage

Build and run with Cargo:

```bash
cargo run --release
```

### Command-Line Options

- `--prompt <TEXT>`: The starting string for text generation (default: "Kim Jong Il of North Korea woke up one morning to find")
- `--numtokens <NUMBER>`: The number of tokens to output (default: 100)
- `--showcandidates <NUMBER>`: The number of top candidate tokens to show (default: 0)

### Examples

Generate text with a custom prompt:

```bash
cargo run --release -- --prompt "Once upon a time" --numtokens 50
```

Show the top 5 candidate tokens during generation with their probabilities. This is useful for understanding the model's decision-making process and seeing which alternatives it considers. The actual output is selected using top-p sampling from the probability distribution.

```bash
cargo run --release -- --prompt "The future of AI is" --showcandidates 5
```

## Testing

Run tests with:

```bash
cargo test
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
