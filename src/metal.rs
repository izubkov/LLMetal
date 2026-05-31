#[derive(Debug, Clone)]
pub struct MetalDevice {
    pub name: String,
}

pub fn preferred_device() -> MetalDevice {
    MetalDevice {
        name: "Apple Metal device 0".to_string(),
    }
}
