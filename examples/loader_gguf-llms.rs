use std::fs::File;
use std::result::Result;

use gguf_llms::{*, Result as GgufResult};

fn main() -> GgufResult<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <path_to_gguf_file>", args[0]);
        std::process::exit(1);
    }

    let mut file = File::open(&args[1])?;

    // Parse file header
    let header = GgufHeader::parse(&mut file)?;

    // Read metadata
    let metadata = GgufReader::read_metadata(&mut file, header.n_kv)?;

    // Extract model configuration
    let config = extract_model_config(&metadata)?;

    // Read tensor information
    let tensor_infos = TensorLoader::read_tensor_info(&mut file, header.n_tensors)?;

    // Get tensor data start position
    let tensor_data_start = TensorLoader::get_tensor_data_start(&mut file)?;

    // Load all tensors
    let tensors = TensorLoader::load_all_tensors(&mut file, &tensor_infos, tensor_data_start)?;

    // Build structured model
    let model = ModelBuilder::new(tensors, config).build()?;

    println!("Loaded {} layers", model.num_layers());
    Ok(())
}