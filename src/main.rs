// A GPT-2 text generation tool in pure Rust

use clap::Parser;
use gpt_encoder::Encoder;
use infer_gpt2::TokenCandidate;
use std::io::{self, Write};
mod infer_gpt2;
mod load_gpt2;

#[derive(Parser, Debug)]
#[clap(about = "Generate text using GPT-2")]
struct Args {
    /// The starting string for text generation
    #[clap(
        long,
        default_value = "Kim Jong Il of North Korea woke up one morning to find"
    )]
    prompt: String,
    /// The number of tokens to output
    #[clap(long, default_value = "100")]
    numtokens: usize,
    /// The number of top candidate tokens to show  
    #[clap(long, default_value = "0")]
    showcandidates: usize,
}

fn print_token(encoder: &Encoder, token: u64) {
    let text = encoder.decode(vec![token]);
    print!("{}", text);
    io::stdout().flush().unwrap();
}

const RED_CITATION_CHAR: &str = "\x1b[31m\"\x1b[0m";
const YELLOW_CITATION_CHAR: &str = "\x1b[33m\"\x1b[0m";

fn print_token_fixed_width(encoder: &Encoder, token: u64) {
    let text = encoder.decode(vec![token]);
    let text_no_newlines = format!("{}{}{}", YELLOW_CITATION_CHAR, text.replace("\n", "<newline>"), YELLOW_CITATION_CHAR);
    print!("{:<34}", text_no_newlines);
}

fn print_token_candidate(encoder: &Encoder, c: TokenCandidate) {
    let text = encoder.decode(vec![c.token_number]);
    let text_no_newlines = text.replace("\n", "<newline>");
    print!(" {}{}{}:{:1.3},", RED_CITATION_CHAR, text_no_newlines, RED_CITATION_CHAR,c.probability);
}

fn main() {
    let args = Args::parse();

    let mut model = load_gpt2::build_from_checkpoint(load_gpt2::DEFAULT_CHECKPOINT_PATH).unwrap();
    println!("Model built successfully\n");
    let sequence_len = 128;

    let mut prompt = args.prompt;
    if prompt.is_empty() {
        prompt = "\n".to_string();
    }

    let mut encoder = gpt_encoder::Encoder::new();
    let mut tokens = encoder.encode(prompt.to_string());
    for token in &tokens {
        print_token(&encoder, *token);
    }
    if args.showcandidates > 0 {
        print!("\n\n");
    }

    for _ in 0..args.numtokens {
        let (token, candidates) =
            infer_gpt2::infer(&mut model, &encoder, &tokens, sequence_len, 0.65);

        // When we reach max context, start trimming it from the start
        if tokens.len() >= sequence_len {
            tokens.remove(0);
        }

        tokens.push(token);

        if args.showcandidates == 0 {
            print_token(&encoder, token);
        } else {
            print_token_fixed_width(&encoder, token);

            for (i, candidate) in candidates.iter().enumerate() {
                if i >= args.showcandidates {
                    break;
                }
                print_token_candidate(&encoder, *candidate);
            }
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // Infer some tokens using a very low top_p, so that the result is predictable
    fn test_predictable_generation() {
        let mut model = load_gpt2::build_from_checkpoint(load_gpt2::DEFAULT_CHECKPOINT_PATH).unwrap();
        let sequence_len = 64;
        let prompt = "To be or not to be, that".to_string();
        let mut encoder = gpt_encoder::Encoder::new();
        let mut tokens = encoder.encode(prompt);

        let (token1, _) = infer_gpt2::infer(&mut model, &encoder, &tokens, sequence_len, 0.0);
        tokens.push(token1);
        print_token(&encoder, token1);

        let (token2, _) = infer_gpt2::infer(&mut model, &encoder, &tokens, sequence_len, 0.0);
        tokens.push(token2);
        print_token(&encoder, token2);

        let (token3, _) = infer_gpt2::infer(&mut model, &encoder, &tokens, sequence_len, 0.0);
        tokens.push(token3);
        print_token(&encoder, token3);

        assert_eq!(token1, 338); // "'s"
        assert_eq!(token2, 644); // " what"
        assert_eq!(token3, 314); // " I"

        println!("second token: \"{}\"", encoder.decode(vec![token2]));
    }
}
