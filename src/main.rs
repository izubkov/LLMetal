use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

// GGUF file format constants
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"
const GGUF_VERSION: u32 = 3;

#[derive(Debug, Clone)]
pub enum GGUFValueType {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

#[derive(Debug, Clone)]
pub enum GGUFValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
}

impl GGMLType {
    fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(GGMLType::F32),
            1 => Some(GGMLType::F16),
            2 => Some(GGMLType::Q4_0),
            3 => Some(GGMLType::Q4_1),
            6 => Some(GGMLType::Q5_0),
            7 => Some(GGMLType::Q5_1),
            8 => Some(GGMLType::Q8_0),
            9 => Some(GGMLType::Q8_1),
            10 => Some(GGMLType::Q2_K),
            11 => Some(GGMLType::Q3_K),
            12 => Some(GGMLType::Q4_K),
            13 => Some(GGMLType::Q5_K),
            14 => Some(GGMLType::Q6_K),
            15 => Some(GGMLType::Q8_K),
            _ => None,
        }
    }

    fn type_size(&self) -> usize {
        match self {
            GGMLType::F32 => 4,
            GGMLType::F16 => 2,
            GGMLType::Q4_0 => 18, // 16 + 2 for scale
            GGMLType::Q4_1 => 20, // 16 + 2 + 2 for scale and min
            GGMLType::Q5_0 => 22, // 20 + 2 for scale
            GGMLType::Q5_1 => 24, // 20 + 2 + 2 for scale and min
            GGMLType::Q8_0 => 34, // 32 + 2 for scale
            GGMLType::Q8_1 => 36, // 32 + 2 + 2 for scale and min
            _ => 32, // Default for K-quants
        }
    }
}

#[derive(Debug)]
pub struct GGUFTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: GGMLType,
    pub offset: u64,
    pub size: usize,
}

#[derive(Debug)]
pub struct GGUFHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

pub struct ModelLoader {
    pub header: GGUFHeader,
    pub metadata: HashMap<String, GGUFValue>,
    pub tensors: HashMap<String, GGUFTensorInfo>,
    pub file: File,
    pub data_offset: u64,
}

impl ModelLoader {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(&file);
        
        // Read header
        let header = Self::read_header(&mut reader)?;
        
        // Read metadata
        let metadata = Self::read_metadata(&mut reader, header.metadata_kv_count)?;
        
        // Read tensor info
        let tensors = Self::read_tensor_info(&mut reader, header.tensor_count)?;
        
        // Calculate data offset (aligned to 32 bytes)
        let current_pos = reader.stream_position()?;
        let data_offset = (current_pos + 31) & !31;
        
        Ok(ModelLoader {
            header,
            metadata,
            tensors,
            file,
            data_offset,
        })
    }
    
    fn read_header(reader: &mut BufReader<&File>) -> Result<GGUFHeader, Box<dyn std::error::Error>> {
        let mut buffer = [0u8; 4];
        
        reader.read_exact(&mut buffer)?;
        let magic = u32::from_le_bytes(buffer);
        
        if magic != GGUF_MAGIC {
            return Err("Invalid GGUF magic number".into());
        }
        
        reader.read_exact(&mut buffer)?;
        let version = u32::from_le_bytes(buffer);
        
        let mut buffer_u64 = [0u8; 8];
        reader.read_exact(&mut buffer_u64)?;
        let tensor_count = u64::from_le_bytes(buffer_u64);
        
        reader.read_exact(&mut buffer_u64)?;
        let metadata_kv_count = u64::from_le_bytes(buffer_u64);
        
        Ok(GGUFHeader {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        })
    }
    
    fn read_metadata(
        reader: &mut BufReader<&File>,
        kv_count: u64,
    ) -> Result<HashMap<String, GGUFValue>, Box<dyn std::error::Error>> {
        let mut metadata = HashMap::new();
        
        for _ in 0..kv_count {
            let key = Self::read_string(reader)?;
            let value_type = Self::read_u32(reader)?;
            let value = Self::read_value(reader, value_type)?;
            metadata.insert(key, value);
        }
        
        Ok(metadata)
    }
    
    fn read_tensor_info(
        reader: &mut BufReader<&File>,
        tensor_count: u64,
    ) -> Result<HashMap<String, GGUFTensorInfo>, Box<dyn std::error::Error>> {
        let mut tensors = HashMap::new();
        let mut offset = 0u64;
        
        for _ in 0..tensor_count {
            let name = Self::read_string(reader)?;
            
            let n_dimensions = Self::read_u32(reader)? as usize;
            let mut dimensions = Vec::with_capacity(n_dimensions);
            
            for _ in 0..n_dimensions {
                dimensions.push(Self::read_u64(reader)?);
            }
            
            let ggml_type_raw = Self::read_u32(reader)?;
            let ggml_type = GGMLType::from_u32(ggml_type_raw)
                .ok_or_else(|| format!("Unknown GGML type: {}", ggml_type_raw))?;
            
            let _tensor_offset = Self::read_u64(reader)?; // This is relative to data section
            
            // Calculate tensor size
            let element_count: u64 = dimensions.iter().product();
            let type_size = ggml_type.type_size();
            let size = match ggml_type {
                GGMLType::Q4_0 | GGMLType::Q4_1 | GGMLType::Q5_0 | GGMLType::Q5_1 |
                GGMLType::Q8_0 | GGMLType::Q8_1 => {
                    // Quantized types are block-based
                    let block_size = 32; // Most quant types use 32-element blocks
                    let blocks = (element_count + block_size - 1) / block_size;
                    blocks as usize * type_size
                }
                _ => element_count as usize * type_size,
            };
            
            let tensor_info = GGUFTensorInfo {
                name: name.clone(),
                dimensions,
                ggml_type,
                offset,
                size,
            };
            
            tensors.insert(name, tensor_info);
            offset += size as u64;
        }
        
        Ok(tensors)
    }
    
    fn read_value(
        reader: &mut BufReader<&File>,
        value_type: u32,
    ) -> Result<GGUFValue, Box<dyn std::error::Error>> {
        match value_type {
            0 => Ok(GGUFValue::UInt8(Self::read_u8(reader)?)),
            1 => Ok(GGUFValue::Int8(Self::read_i8(reader)?)),
            2 => Ok(GGUFValue::UInt16(Self::read_u16(reader)?)),
            3 => Ok(GGUFValue::Int16(Self::read_i16(reader)?)),
            4 => Ok(GGUFValue::UInt32(Self::read_u32(reader)?)),
            5 => Ok(GGUFValue::Int32(Self::read_i32(reader)?)),
            6 => Ok(GGUFValue::Float32(Self::read_f32(reader)?)),
            7 => Ok(GGUFValue::Bool(Self::read_u8(reader)? != 0)),
            8 => Ok(GGUFValue::String(Self::read_string(reader)?)),
            9 => {
                let array_type = Self::read_u32(reader)?;
                let array_len = Self::read_u64(reader)? as usize;
                let mut array = Vec::with_capacity(array_len);
                for _ in 0..array_len {
                    array.push(Self::read_value(reader, array_type)?);
                }
                Ok(GGUFValue::Array(array))
            }
            10 => Ok(GGUFValue::UInt64(Self::read_u64(reader)?)),
            11 => Ok(GGUFValue::Int64(Self::read_i64(reader)?)),
            12 => Ok(GGUFValue::Float64(Self::read_f64(reader)?)),
            _ => Err(format!("Unknown value type: {}", value_type).into()),
        }
    }
    
    // Helper read functions
    fn read_u8(reader: &mut BufReader<&File>) -> Result<u8, std::io::Error> {
        let mut buffer = [0u8; 1];
        reader.read_exact(&mut buffer)?;
        Ok(buffer[0])
    }
    
    fn read_i8(reader: &mut BufReader<&File>) -> Result<i8, std::io::Error> {
        Ok(Self::read_u8(reader)? as i8)
    }
    
    fn read_u16(reader: &mut BufReader<&File>) -> Result<u16, std::io::Error> {
        let mut buffer = [0u8; 2];
        reader.read_exact(&mut buffer)?;
        Ok(u16::from_le_bytes(buffer))
    }
    
    fn read_i16(reader: &mut BufReader<&File>) -> Result<i16, std::io::Error> {
        let mut buffer = [0u8; 2];
        reader.read_exact(&mut buffer)?;
        Ok(i16::from_le_bytes(buffer))
    }
    
    fn read_u32(reader: &mut BufReader<&File>) -> Result<u32, std::io::Error> {
        let mut buffer = [0u8; 4];
        reader.read_exact(&mut buffer)?;
        Ok(u32::from_le_bytes(buffer))
    }
    
    fn read_i32(reader: &mut BufReader<&File>) -> Result<i32, std::io::Error> {
        let mut buffer = [0u8; 4];
        reader.read_exact(&mut buffer)?;
        Ok(i32::from_le_bytes(buffer))
    }
    
    fn read_f32(reader: &mut BufReader<&File>) -> Result<f32, std::io::Error> {
        let mut buffer = [0u8; 4];
        reader.read_exact(&mut buffer)?;
        Ok(f32::from_le_bytes(buffer))
    }
    
    fn read_u64(reader: &mut BufReader<&File>) -> Result<u64, std::io::Error> {
        let mut buffer = [0u8; 8];
        reader.read_exact(&mut buffer)?;
        Ok(u64::from_le_bytes(buffer))
    }
    
    fn read_i64(reader: &mut BufReader<&File>) -> Result<i64, std::io::Error> {
        let mut buffer = [0u8; 8];
        reader.read_exact(&mut buffer)?;
        Ok(i64::from_le_bytes(buffer))
    }
    
    fn read_f64(reader: &mut BufReader<&File>) -> Result<f64, std::io::Error> {
        let mut buffer = [0u8; 8];
        reader.read_exact(&mut buffer)?;
        Ok(f64::from_le_bytes(buffer))
    }
    
    fn read_string(reader: &mut BufReader<&File>) -> Result<String, Box<dyn std::error::Error>> {
        let len = Self::read_u64(reader)? as usize;
        let mut buffer = vec![0u8; len];
        reader.read_exact(&mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
    
    // Load tensor data
    pub fn load_tensor(&mut self, name: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let tensor_info = self.tensors.get(name)
            .ok_or_else(|| format!("Tensor '{}' not found", name))?;
        
        let mut file = &self.file;
        file.seek(SeekFrom::Start(self.data_offset + tensor_info.offset))?;
        
        let mut buffer = vec![0u8; tensor_info.size];
        file.read_exact(&mut buffer)?;
        
        Ok(buffer)
    }
    
    // Get model architecture info
    pub fn get_architecture(&self) -> Option<&str> {
        if let Some(GGUFValue::String(arch)) = self.metadata.get("general.architecture") {
            Some(arch)
        } else {
            None
        }
    }
    
    pub fn get_context_length(&self) -> Option<u64> {
        if let Some(GGUFValue::UInt32(ctx_len)) = self.metadata.get("llama.context_length") {
            Some(*ctx_len as u64)
        } else if let Some(GGUFValue::UInt64(ctx_len)) = self.metadata.get("llama.context_length") {
            Some(*ctx_len)
        } else {
            None
        }
    }
    
    pub fn get_embedding_length(&self) -> Option<u64> {
        if let Some(GGUFValue::UInt32(emb_len)) = self.metadata.get("llama.embedding_length") {
            Some(*emb_len as u64)
        } else if let Some(GGUFValue::UInt64(emb_len)) = self.metadata.get("llama.embedding_length") {
            Some(*emb_len)
        } else {
            None
        }
    }
    
    pub fn get_layer_count(&self) -> Option<u64> {
        if let Some(GGUFValue::UInt32(layers)) = self.metadata.get("llama.block_count") {
            Some(*layers as u64)
        } else if let Some(GGUFValue::UInt64(layers)) = self.metadata.get("llama.block_count") {
            Some(*layers)
        } else {
            None
        }
    }

    fn format_array(&self, arr: &Vec<GGUFValue>) -> String {
        if arr.len() > 3 {
            format!("[{:?}, {:?}, ... {:?}", arr.get(0), arr.get(1), arr.last())
        } else {
            format!("{:?}", arr)
        }
    }
    
    // Print model info
    pub fn print_info(&self) {
        println!("=== Model Information ===");
        println!("Architecture: {:?}", self.get_architecture());
        println!("Context Length: {:?}", self.get_context_length());
        println!("Embedding Length: {:?}", self.get_embedding_length());
        println!("Layer Count: {:?}", self.get_layer_count());
        println!("Tensor Count: {}", self.header.tensor_count);
        
        println!("\n=== Tensors ===");
        for (name, info) in &self.tensors {
            println!("{}: {:?} {:?}", name, info.dimensions, info.ggml_type);
        }
        
        println!("\n=== Metadata ===");
        for (key, value) in &self.metadata {
            match value {
                GGUFValue::String(s) => println!("{}: {}", key, s),
                GGUFValue::UInt32(n) => println!("{}: {}", key, n),
                GGUFValue::UInt64(n) => println!("{}: {}", key, n),
                GGUFValue::Float32(f) => println!("{}: {}", key, f),
                GGUFValue::Bool(b) => println!("{}: {}", key, b),
                GGUFValue::Array(a) => println!("{}: {:?}", key, &self.format_array(a)),
                _ => println!("{}: {:?}", key, value),
            }
        }
    }
}

// Example usage
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    dbg!(&args);
    if args.len() != 2 {
        eprintln!("Usage: {} <path_to_gguf_file>", args[0]);
        std::process::exit(1);
    }
    
    let mut loader = ModelLoader::new(&args[1])?;
    loader.print_info();
    
    // Example: Load the token embeddings tensor
    if loader.tensors.contains_key("token_embd.weight") {
        println!("\nLoading token embeddings...");
        let embeddings = loader.load_tensor("token_embd.weight")?;
        println!("Loaded {} bytes of token embeddings", embeddings.len());
    }
    
    Ok(())
}