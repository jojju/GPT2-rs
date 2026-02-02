use std::{
    fs::File,
    io::{Error, Read},
    mem,
    process::exit,
    slice,
};

pub const DEFAULT_CHECKPOINT_PATH: &str = "gpt2_small_124M.bin";

// Helper function for reading from a file into a buffer of type T
fn read_to_buffer<T: Copy>(file: &mut File, buffer: &mut [T]) -> Result<(), Error> {
    // Get the total number of bytes to read.
    let buffer_size_bytes = mem::size_of_val(buffer);

    if buffer_size_bytes == 0 {
        return Ok(());
    }

    // Get a mutable byte slice view of the T slice.
    let byte_buffer =
        unsafe { slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut u8, buffer_size_bytes) };

    // Read directly into the byte view of the target buffer.
    file.read_exact(byte_buffer)?;

    Ok(())
}

#[derive(Debug, Clone)]
pub struct GPT2Config {
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub padded_vocab_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub channels: usize,
}

// A struct containing all the parameters of the GPT-2 small model
#[derive(Debug)]
pub struct GPT2ParamTensors {
    // V: vocabulary size
    // C: "channels" / embedding dimension
    // maxT: max sequence length
    // L: number of layers

    // Word Token Embeddings: Maps vocabulary indices to embedding vectors.
    pub token_embedding_weights: Vec<f32>, // (V, C)
    // Word Positional Embeddings: Maps sequence positions to embedding vectors.
    pub positional_embedding_weights: Vec<f32>, // (maxT, C)

    // --- Parameters for each Transformer Layer (repeated L times) ---

    // Layer Normalization 1 (before self-attention) weights
    pub attention_layer_norm_weights: Vec<f32>, // (L, C)
    // Layer Normalization 1 (before self-attention) biases
    pub attention_layer_norm_biases: Vec<f32>, // (L, C)

    // Self-Attention Query, Key, Value projection weights
    pub attention_qkv_weights: Vec<f32>, // (L, 3*C, C)
    // Self-Attention Query, Key, Value projection biases
    pub attention_qkv_biases: Vec<f32>, // (L, 3*C)

    // Self-Attention output projection weights
    pub attention_output_projection_weights: Vec<f32>, // (L, C, C)
    // Self-Attention output projection biases
    pub attention_output_projection_biases: Vec<f32>, // (L, C)

    // Layer Normalization 2 (before feed-forward network) weights
    pub feed_forward_layer_norm_weights: Vec<f32>, // (L, C)
    // Layer Normalization 2 (before feed-forward network) biases
    pub feed_forward_layer_norm_biases: Vec<f32>, // (L, C)

    // Feed-Forward Network: first linear layer (expansion) weights
    pub feed_forward_expansion_weights: Vec<f32>, // (L, 4*C, C)
    // Feed-Forward Network: first linear layer (expansion) biases
    pub feed_forward_expansion_biases: Vec<f32>, // (L, 4*C)

    // Feed-Forward Network: second linear layer (projection) weights
    pub feed_forward_projection_weights: Vec<f32>, // (L, C, 4*C)
    // Feed-Forward Network: second linear layer (projection) biases
    pub feed_forward_projection_biases: Vec<f32>, // (L, C)

    // --- End of per-layer parameters ---

    // Final Layer Normalization weights
    pub final_layer_norm_weights: Vec<f32>, // (C)
    // Final Layer Normalization biases
    pub final_layer_norm_biases: Vec<f32>, // (C)
}

impl GPT2ParamTensors {
    /// Creates a new struct with all its vectors initialized
    pub fn new_empty(config: &GPT2Config) -> Self {
        let vp = config.padded_vocab_size; // Vocabulary Size (Padded)
        let c = config.channels; // Embedding Dimension / Number of Channels
        let max_t = config.max_seq_len; // Maximum Sequence Length
        let l = config.num_layers; // Number of Transformer Layers

        GPT2ParamTensors {
            token_embedding_weights: vec![0.0; vp * c],
            positional_embedding_weights: vec![0.0; max_t * c],
            attention_layer_norm_weights: vec![0.0; l * c],
            attention_layer_norm_biases: vec![0.0; l * c],
            attention_qkv_weights: vec![0.0; l * 3 * c * c],
            attention_qkv_biases: vec![0.0; l * 3 * c],
            attention_output_projection_weights: vec![0.0; l * c * c],
            attention_output_projection_biases: vec![0.0; l * c],
            feed_forward_layer_norm_weights: vec![0.0; l * c],
            feed_forward_layer_norm_biases: vec![0.0; l * c],
            feed_forward_expansion_weights: vec![0.0; l * 4 * c * c],
            feed_forward_expansion_biases: vec![0.0; l * 4 * c],
            feed_forward_projection_weights: vec![0.0; l * c * 4 * c],
            feed_forward_projection_biases: vec![0.0; l * c],
            final_layer_norm_weights: vec![0.0; c],
            final_layer_norm_biases: vec![0.0; c],
        }
    }

    /// Calculates the total number of floating-point parameters in the struct.
    pub fn total_len(&self) -> usize {
        self.token_embedding_weights.len()
            + self.positional_embedding_weights.len()
            + self.attention_layer_norm_weights.len()
            + self.attention_layer_norm_biases.len()
            + self.attention_qkv_weights.len()
            + self.attention_qkv_biases.len()
            + self.attention_output_projection_weights.len()
            + self.attention_output_projection_biases.len()
            + self.feed_forward_layer_norm_weights.len()
            + self.feed_forward_layer_norm_biases.len()
            + self.feed_forward_expansion_weights.len()
            + self.feed_forward_expansion_biases.len()
            + self.feed_forward_projection_weights.len()
            + self.feed_forward_projection_biases.len()
            + self.final_layer_norm_weights.len()
            + self.final_layer_norm_biases.len()
    }
}

// Read parameters from the model file
pub fn load_parameters_from_file(
    model_file: &mut File,
    params: &mut GPT2ParamTensors, // Now takes a mutable reference to an existing struct
) -> Result<(), Error> {
    // The struct `params` is already created and its vectors are allocated
    // to the correct sizes. We just need to read the data into them.
    read_to_buffer(model_file, params.token_embedding_weights.as_mut_slice())?;
    read_to_buffer(
        model_file,
        params.positional_embedding_weights.as_mut_slice(),
    )?;
    read_to_buffer(
        model_file,
        params.attention_layer_norm_weights.as_mut_slice(),
    )?;
    read_to_buffer(
        model_file,
        params.attention_layer_norm_biases.as_mut_slice(),
    )?;
    read_to_buffer(model_file, params.attention_qkv_weights.as_mut_slice())?;
    read_to_buffer(model_file, params.attention_qkv_biases.as_mut_slice())?;
    read_to_buffer(
        model_file,
        params.attention_output_projection_weights.as_mut_slice(),
    )?;
    read_to_buffer(
        model_file,
        params.attention_output_projection_biases.as_mut_slice(),
    )?;
    read_to_buffer(
        model_file,
        params.feed_forward_layer_norm_weights.as_mut_slice(),
    )?;
    read_to_buffer(
        model_file,
        params.feed_forward_layer_norm_biases.as_mut_slice(),
    )?;
    read_to_buffer(
        model_file,
        params.feed_forward_expansion_weights.as_mut_slice(),
    )?;
    read_to_buffer(
        model_file,
        params.feed_forward_expansion_biases.as_mut_slice(),
    )?;
    read_to_buffer(
        model_file,
        params.feed_forward_projection_weights.as_mut_slice(),
    )?;
    read_to_buffer(
        model_file,
        params.feed_forward_projection_biases.as_mut_slice(),
    )?;
    read_to_buffer(model_file, params.final_layer_norm_weights.as_mut_slice())?;
    read_to_buffer(model_file, params.final_layer_norm_biases.as_mut_slice())?;

    Ok(())
}
#[derive(Debug)]
pub struct GPT2 {
    pub config: GPT2Config,
    pub params: GPT2ParamTensors,
}

pub fn build_from_checkpoint(checkpoint_path: &str) -> Result<GPT2, Error> {
    // read in model from a checkpoint file
    let mut model_file = File::open(checkpoint_path)?;
    let mut model_header: [i32; 256] = [0; 256];
    read_to_buffer(&mut model_file, &mut model_header)?;
    if model_header[0] != 20240326 {
        eprintln!("Bad magic in model file");
        exit(1);
    }
    if model_header[1] != 3 {
        eprintln!("Bad version in model file");
        exit(1);
    }

    // Read hyperparameters from model
    let max_seq_len = model_header[2] as usize;
    let vocab_size = model_header[3] as usize;
    let layers = model_header[4] as usize;
    let heads = model_header[5] as usize;
    let channels = model_header[6] as usize;
    let padded_vocab_size = model_header[7] as usize;

    let config = GPT2Config {
        max_seq_len,
        vocab_size,
        padded_vocab_size,
        num_layers: layers,
        num_heads: heads,
        channels,
    };

    println!("Loading GPT-2 model");
    println!("Max sequence length: {}", max_seq_len);
    println!("Vocabulary size: {}", vocab_size);
    println!("Padded vocabulary size: {}", padded_vocab_size);
    println!("Layers: {}", layers);
    println!("Attention heads: {}", heads);
    println!("Channels: {}", channels);

    let mut params = GPT2ParamTensors::new_empty(&config);

    println!("Parameters: {}", params.total_len());

    load_parameters_from_file(&mut model_file, &mut params)?;

    Ok(GPT2 {
        config: config.clone(),
        params,
    })
}

#[test]
// Just a quick test that some data has been read correctly.
fn test_gpt2_build_from_checkpoint() {
    let model = build_from_checkpoint(DEFAULT_CHECKPOINT_PATH).unwrap();
    let expected_wpe: [f32; 10] = [
        -0.01882072,
        -0.1974186,
        0.004026725,
        0.011346859,
        0.06382412,
        -0.10501328,
        0.036937047,
        -0.16802956,
        -0.04911101,
        -0.05646128,
    ];
    for i in 0..10 {
        assert_eq!(
            model.params.positional_embedding_weights[i],
            expected_wpe[i]
        );
    }
}
