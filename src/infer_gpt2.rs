use crate::load_gpt2;
use gpt_encoder;

// Return embeddings corresponding to provided tokens, plus position encodings.
// Token embeddings should be arranged in order, with the index corresponding to the token.
fn add_embeddings_and_position_encodings(
    out: &mut Vec<f32>,
    tokens: &[u64],
    token_embeddings: &Vec<f32>,
    positional_encodings: &Vec<f32>,
    embedding_size: usize,
) {
    for (i, tok) in tokens.iter().enumerate() {
        let start_idx = *tok as usize * embedding_size;
        let embedding = &token_embeddings[start_idx..start_idx + embedding_size];
        let start_idx = i * embedding_size;
        let end_idx = start_idx + (embedding_size); //- 1);
        let position_encoding = &positional_encodings[start_idx..end_idx];
        let mut k = 0;
        for j in start_idx..end_idx {
            // Position encodings are simply added to the embedding
            out[j] = embedding[k] + position_encoding[k];
            k = k + 1;
        }
    }
}

// Normalize each slice in inp of size channels
fn layer_normalize(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    seq_len: usize,  // Number of embeddings in input
    channels: usize, // "Channels", i.e. weights per embedding
) {
    // Just a small number used to prevent division by zero and improve numerical stability
    let epsilon = 1e-5f32;

    for seq_idx in 0..seq_len {
        // Seek to the right input position
        let x = &inp[seq_idx * channels..];

        // Calculate the mean
        let mut sum = 0.0f32;
        for i in 0..channels {
            sum += x[i];
        }
        let mean = sum / channels as f32;

        // Calculate the variance
        let mut v = 0.0f32;
        for i in 0..channels {
            let xshift = x[i] - mean;
            v += xshift * xshift;
        }
        let variance = v / channels as f32;

        // Calculate inverse standard deviation
        let inv_std = 1.0f32 / (variance + epsilon).sqrt();

        // Seek to the right position in out
        let out_bt = &mut out[seq_idx * channels..];

        // Normalize, scale, and shift
        for i in 0..channels {
            // Subtract the mean -> make the new mean 0
            // Multiply by 1 / std-dev -> make the new std-dev 1
            let normalized = inv_std * (x[i] - mean);
            // Apply the (learned) weight and bias
            out_bt[i] = normalized * weight[i] + bias[i];
        }
    }
}

// fn matmul_forward_simple(
//     out: &mut [f32],
//     input: &[f32],
//     weight: &[f32],
//     bias: Option<&[f32]>,
//     t: usize,  // "time" i.e. number of tokens
//     c: usize,  // "channels", i.e. number of weights per embedding
//     oc: usize, // "output channels"
// ) {
//     // for each token
//     for t_idx in 0..t {
//         // for each output channel
//         for o in 0..oc {
//             let mut val = if let Some(bias_val) = bias {
//                 bias_val[o]
//             } else {
//                 0.0
//             };
//             // For each input and weight, multiply them. Then add bias.
//             for i in 0..c {
//                 val += input[t_idx * c + i] * weight[o * c + i];
//             }
//             out[t_idx * oc + o] = val;
//         }
//     }
// }

use rayon::prelude::*;

fn matmul_with_bias(
    out: &mut [f32],
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    num_channels: usize, // I.e. number of weights per embedding
    num_output_channels: usize,
) {
    // Parallelize over tokens and output channels. We use par_chunks_mut()
    // on the output to get a part of it.
    out.par_chunks_mut(num_output_channels)
        .enumerate()
        .for_each(|(token_idx, out_chunk)| {
            // out_chunk is a mutable slice of size output_channels corresponding to all of
            // the output for token token_idx
            for out_chan in 0..num_output_channels {
                // Get the bias, or 0.0 if there is none, as the starting value
                let mut val = bias.map_or(0.0, |b| b[out_chan]);

                // For each channel, add up all the [input * weight]
                for i in 0..num_channels {
                    val +=
                        input[token_idx * num_channels + i] * weight[out_chan * num_channels + i];
                }
                out_chunk[out_chan] = val;
            }
        });
}

// Scaled dot-product attention
// inp contains query, key and value vectors, calculated in a previous step
// Data layout of inp:
// [Q(tok1), K(tok1), V(tok1), Q(tok2), K(tok2), V(tok2), ..., Q(tokT), K(tokT), V(tokT)]
// where each Q(tok_i), K(tok_i), V(tok_i) is a vector of size c.
fn attention(out: &mut [f32], inp: &[f32], c: usize, n_heads: usize) {
    let c3 = c * 3;
    // Each head works on only a part of the input vectors of size c
    let head_size = c / n_heads;
    // A scaling factor to reduce the dot product of the query and key vectors
    // This is done to get values suitable for softmax
    let scale = 1.0 / (head_size as f32).sqrt();

    // Calculate t_len (sequence length) from inp
    let t_len = (inp.len() / c3) as usize;
    // Define preatt and att based on t, n_heads
    let mut preatt = vec![0.0f32; n_heads * t_len * t_len];
    let mut att = vec![0.0f32; n_heads * t_len * t_len];

    // For each token in the sequence
    for t_idx in 0..t_len {
        // For each head in the model
        for head_idx in 0..n_heads {
            // The query for the current token and head
            let query_offset = t_idx * c3 + head_idx * head_size;

            let preatt_offset = head_idx * t_len * t_len + t_idx * t_len;
            let att_offset = head_idx * t_len * t_len + t_idx * t_len;

            // For all tokens, calculate <query dot key> (and remember the maximum value)
            // This tells us how much attention to pay to this token
            let mut maxval = -10000.0f32;
            // We only need to calculate the attention for the current token and all previous tokens
            // -- we never care about future tokens
            for t_curr in 0..=t_idx {
                // The key for the current token
                // +c because the key has second position in "qkv" (query, key, value)
                let key_offset = t_curr * c3 + c + head_idx * head_size;

                // Calculate dot product between query and key vectors
                let mut val = 0.0f32;
                for i in 0..head_size {
                    val += inp[query_offset + i] * inp[key_offset + i];
                }
                val *= scale;
                if val > maxval {
                    maxval = val;
                }
                preatt[preatt_offset + t_curr] = val;
            }

            // Run softmax on the <query dot key> logits
            softmax(&mut att[att_offset..], &preatt[preatt_offset..], t_idx + 1);

            let out_offset = t_idx * c + head_idx * head_size;
            // Set starting values to 0
            for i in 0..head_size {
                out[out_offset + i] = 0.0;
            }

            // Accumulate weighted values for all previous tokens into the output
            for t_curr in 0..=t_idx {
                // + c*2 because the value has third position in "qkv" (query, key, value)
                let value_offset = t_curr * c3 + head_idx * head_size + c * 2;
                let att_score = att[att_offset + t_curr];
                for i in 0..head_size {
                    // Add the weighted value belonging to the current embedding
                    out[out_offset + i] += att_score * inp[value_offset + i];
                }
            }
        }
    }
}

// Simply add input1 and input2 into out
fn add_residual(out: &mut [f32], input1: &[f32], input2: &[f32]) {
    assert!(out.len() == input1.len() && out.len() == input2.len());
    for i in 0..input1.len() {
        out[i] = input1[i] + input2[i];
    }
}

use std::f32::consts::PI;
use std::sync::OnceLock;

// A value that is calculated only once
static GELU_SCALING_FACTOR: OnceLock<f32> = OnceLock::new();

// Element-wise GeLU activation function
fn gelu(out: &mut [f32], inp: &[f32]) {
    assert_eq!(inp.len(), out.len());
    let scaling_factor = GELU_SCALING_FACTOR.get_or_init(|| (2.0_f32 / PI).sqrt());
    for i in 0..inp.len() {
        let x = inp[i];
        let cube = 0.044715_f32 * x * x * x;
        out[i] = 0.5_f32 * x * (1.0_f32 + (*scaling_factor * (x + cube)).tanh());
    }
}

use std::f32;

// Output values as probabilities that sum to 1
fn softmax(probs: &mut [f32], logits: &[f32], size: usize) {
    // output: probs are the probabilities that will sum to 1.0
    // logits: the raw inputs

    // maxval is only calculated and subtracted for numerical stability
    let maxval = logits[..size]
        .iter()
        .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));

    let mut sum = 0.0;
    for i in 0..size {
        probs[i] = (logits[i] - maxval).exp();
        sum += probs[i];
    }

    for i in 0..size {
        probs[i] /= sum;
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct TokenCandidate {
    pub token_number: u64,
    pub probability: f32,
}

impl std::fmt::Display for TokenCandidate {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} {} ", self.token_number, self.probability)
    }
}

// Gather the most probable token candidates until the cumulative probability reaches top_p
// The result is sorted by probability (descending)
fn top_p_filtering(probs: &[f32], top_p: f32, vocab_size: usize) -> Vec<TokenCandidate> {
    assert!(probs.len() == vocab_size);
    assert!(top_p >= 0.0 && top_p <= 1.0);

    // Create a TokenCandidate for each token in the vocabulary
    let mut candidates: Vec<TokenCandidate> = probs[0..vocab_size]
        .iter()
        .enumerate()
        .filter(|(_, &p)| p > 0.0 && p.is_finite()) // Only consider valid positive probabilities
        .map(|(index, &value)| TokenCandidate {
            token_number: index as u64,
            probability: value,
        })
        .collect();

    // Sort by probability (descending), then by token_number as a tie-breaker
    candidates.sort_unstable_by(|a, b| {
        b.probability
            .partial_cmp(&a.probability)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.token_number.cmp(&b.token_number))
    });

    // Apply Top-P filtering to the sorted candidates
    let mut cumulative_prob = 0.0;
    let mut nbr_candidates = 0;

    for indexed_value in candidates.iter() {
        // Add this token's probability to the sum. Ensure we always include at least one token.
        cumulative_prob += indexed_value.probability;
        // Set the current number of candidates added
        nbr_candidates += 1;
        // Check if adding this token would exceed the threshold.
        // It's important to include the token that *crosses* the threshold.
        if cumulative_prob >= top_p {
            break; // Stop adding tokens once cumulative probability is met
        }
    }

    // Truncate the vector to keep only the wanted tokens
    candidates.truncate(nbr_candidates);

    // Result now contains the smallest set whose cumulative probability >= top_p
    candidates
}

use rand::Rng;

fn sample_from_probabilities(items: &[TokenCandidate]) -> Option<TokenCandidate> {
    if items.is_empty() {
        return None;
    }

    // Calculate the sum of probabilities.
    let total_probability: f32 = items.iter().map(|x| x.probability).sum();

    if total_probability <= 0.0 || !total_probability.is_finite() {
        return None; // Handle cases with no valid probabilities
    }

    // Generate a random number between 0 and the total probability.
    let mut rng = rand::rng();
    let random_value = rng.random_range(0.0..total_probability);

    // Iterate through the items, accumulating probabilities until we exceed the random value.
    let mut accumulated_probability = 0.0;
    for &item in items {
        accumulated_probability += item.probability;
        if accumulated_probability >= random_value {
            return Some(item);
        }
    }

    panic!(); // Should never reach here
}

#[allow(unused_variables)]
pub fn infer(
    model: &mut load_gpt2::GPT2,
    encoder: &gpt_encoder::Encoder,
    tokens: &Vec<u64>,
    seq_len: usize,
    top_p: f32,
) -> (u64, Vec<TokenCandidate>) {
    let c = model.config.channels; // The size of the token embedding
    let mut encoded = vec![0.0; c * seq_len];

    // Translate all token IDs into vectors using the token embeddings and positional encodings
    add_embeddings_and_position_encodings(
        &mut encoded,
        tokens,
        &model.params.token_embedding_weights,
        &model.params.positional_embedding_weights,
        model.config.channels,
    );

    // Allocate vectors to use for the steps in each layer
    let size = c * seq_len;
    let mut attn_ln_out = vec![0.0; size];
    let mut qkv = vec![0.0; size * 3]; // Will hold the three q, k, v vectors
    let mut attn_out = vec![0.0; size];
    let mut attn_proj = vec![0.0; size];
    let mut attn_residual = vec![0.0; size];
    let mut ff_ln_out = vec![0.0; size];
    let mut ff_expanded = vec![0.0; size * 4];
    let mut ff_activated = vec![0.0; size * 4];
    let mut ff_projected = vec![0.0; size];
    let mut block_output = vec![0.0; size];
    let mut final_norm = vec![0.0; c];

    let position = tokens.len() - 1;

    // For every layer in the model
    for l in 0..model.config.num_layers {
        // Set weights and biases as slices from the right locations in the model vectors

        let layer_offset_c = l * c; // Layer index times channels (c)
        let qkv_w_offset = l * 3 * c * c; // 3 = Q, K, V; each is c x c
        let qkv_b_offset = l * 3 * c; // 3 = Q, K, V; each bias is length c
        let attn_proj_w_offset = l * c * c; // One attention projection matrix of size c x c
        let ff_w_expansion_offset = l * 4 * c * c; // 4 = expansion to 4c; weights are 4c x c
        let ff_b_expansion_offset = l * 4 * c; // 4 = expansion to 4c; bias is length 4c
        let ff_w_projection_offset = l * c * 4 * c; // Projection back to c; weights are c x 4c

        let ln_size = c; // Layer-norm has one vector of length c
        let qkv_w_size = 3 * c * c; // 3 = Q, K, V; each weight matrix is c x c
        let qkv_b_size = 3 * c; // 3 = Q, K, V; each bias is length c
        let attn_proj_w_size = c * c; // One projection matrix of size c x c
        let ff_expansion_w_size = 4 * c * c; // Expand to 4c: weights are 4c x c
        let ff_expansion_b_size = 4 * c; // Expand to 4c: bias is length 4c
        let ff_projection_w_size = c * 4 * c; // Project back: weights are c x 4c

        let layernorm_1_w =
            &model.params.attention_layer_norm_weights[layer_offset_c..layer_offset_c + ln_size];
        let layernorm_1_b =
            &model.params.attention_layer_norm_biases[layer_offset_c..layer_offset_c + ln_size];

        let qkv_w = &model.params.attention_qkv_weights[qkv_w_offset..qkv_w_offset + qkv_w_size];
        let qkv_b = &model.params.attention_qkv_biases[qkv_b_offset..qkv_b_offset + qkv_b_size];

        let attn_projection_w = &model.params.attention_output_projection_weights
            [attn_proj_w_offset..attn_proj_w_offset + attn_proj_w_size];
        let attn_projection_b = &model.params.attention_output_projection_biases
            [layer_offset_c..layer_offset_c + ln_size];

        let layernorm_2_w =
            &model.params.feed_forward_layer_norm_weights[layer_offset_c..layer_offset_c + ln_size];
        let layernorm_2_b =
            &model.params.feed_forward_layer_norm_biases[layer_offset_c..layer_offset_c + ln_size];

        let feed_fw_expansion_w = &model.params.feed_forward_expansion_weights
            [ff_w_expansion_offset..ff_w_expansion_offset + ff_expansion_w_size];
        let feed_fw_expansion_b = &model.params.feed_forward_expansion_biases
            [ff_b_expansion_offset..ff_b_expansion_offset + ff_expansion_b_size];

        let feed_fw_contraction_w = &model.params.feed_forward_projection_weights
            [ff_w_projection_offset..ff_w_projection_offset + ff_projection_w_size];
        let feed_fw_contraction_b =
            &model.params.feed_forward_projection_biases[layer_offset_c..layer_offset_c + ln_size];

        // If this is the first layer, start with the original encodings. Otherwise, start with the
        // output from the last layer.
        let block_input = match l {
            0 => &encoded,
            _ => &block_output,
        };

        // First normalize
        layer_normalize(
            &mut attn_ln_out,
            &block_input,
            &layernorm_1_w,
            &layernorm_1_b,
            seq_len,
            c,
        );

        // Calculate keys, queries and values (qkv)
        matmul_with_bias(&mut qkv, &attn_ln_out, &qkv_w, Some(&qkv_b), c, 3 * c);

        // Carry out the attention mechanism
        attention(&mut attn_out, &qkv, c, model.config.num_heads);

        // Make a new linear projection of the attention output
        matmul_with_bias(
            &mut attn_proj,
            &attn_out,
            &attn_projection_w,
            Some(&attn_projection_b),
            c,
            c,
        );

        // Add back the first vector as a skip connection
        add_residual(&mut attn_residual, &attn_proj, &block_input);

        // Normalize again
        layer_normalize(
            &mut ff_ln_out,
            &attn_residual,
            &layernorm_2_w,
            &layernorm_2_b,
            seq_len,
            c,
        );

        // Matrix multipy, increasing the size * 4
        matmul_with_bias(
            &mut ff_expanded,
            &ff_ln_out,
            feed_fw_expansion_w,
            Some(feed_fw_expansion_b),
            c,
            4 * c,
        );

        // Run a GeLU activation function
        gelu(&mut ff_activated, &ff_expanded);

        // Matrix multipy, decreasing the size again
        matmul_with_bias(
            &mut ff_projected,
            &ff_activated,
            &feed_fw_contraction_w,
            Some(&feed_fw_contraction_b),
            4 * c,
            c,
        );

        // Make a new skip connection
        add_residual(&mut block_output, &ff_projected, &attn_residual);
    }

    // Do a final normalization
    layer_normalize(
        &mut final_norm,
        &block_output[position * c..position * c + c], // Now we only need the vector for the new token
        &model.params.final_layer_norm_weights,
        &model.params.final_layer_norm_biases,
        1,
        c,
    );

    // Calculate the logits (of the same size as the vocabulary)
    let last_embedding = &final_norm;
    let padded_vocab_size = model.config.padded_vocab_size;
    let vocab_size = model.config.vocab_size;
    let mut logits = vec![0.0; padded_vocab_size];
    matmul_with_bias(
        &mut logits,
        &last_embedding,
        &model.params.token_embedding_weights,
        None,
        c,
        padded_vocab_size,
    );

    // Use a hard coded temperature
    for i in 0..logits.len() {
        logits[i] /= 0.9;
    }

    // Do a softmax over the logits
    let mut probs = vec![0.0; vocab_size];
    softmax(&mut probs, &logits, vocab_size);

    // Run top P filtering to select candidate tokens
    let candidates = top_p_filtering(&probs, top_p, vocab_size);

    // Select the final token
    let selected = sample_from_probabilities(&candidates).unwrap().token_number as u64;

    return (selected, candidates);
}

// Functions useful for debugging

#[allow(dead_code)]
fn print_candidates(encoder: &gpt_encoder::Encoder, candidates: &Vec<TokenCandidate>) {
    println!();
    for c in candidates {
        print!(
            "{:1.3} {}, ",
            c.probability,
            encoder.decode(vec![c.token_number as u64])
        );
    }
    println!();
}

#[allow(dead_code)]
fn print_slice(arr: &[f32], text: &str, start: usize, num: usize) {
    print!("{}: ", text);
    for i in 0..num {
        print!("{:.3} ", arr[start + i]);
    }
    print!("\n");
}
