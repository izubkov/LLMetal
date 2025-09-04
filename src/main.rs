use gguf_rs::get_gguf_container;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <path_to_gguf_file>", args[0]);
        std::process::exit(1);
    }

    let mut container = get_gguf_container(&args[1])?;
    let model = container.decode()?;
    println!("Model Family: {}", model.model_family());
    println!("Number of Parameters: {}", model.model_parameters());
    println!("File Type: {}", model.file_type());
    println!("Number of Tensors: {}", model.num_tensor());
    Ok(())
}