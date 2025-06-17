use metal::*;
use std::mem;

fn main() {
    // Initialize Metal device and command queue
    let device = Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();

    println!("{:?}", device);

    // Create buffers
    let n = 1024;
    let a = vec![1.0f32; n];
    let b = vec![2.0f32; n];
    let mut c = vec![0.0f32; n];

    let a_buffer = device.new_buffer_with_data(a.as_ptr() as *const _, (n * mem::size_of::<f32>()) as u64, MTLResourceOptions::StorageModeShared);
    let b_buffer = device.new_buffer_with_data(b.as_ptr() as *const _, (n * mem::size_of::<f32>()) as u64, MTLResourceOptions::StorageModeShared);
    let c_buffer = device.new_buffer_with_data(c.as_mut_ptr() as *mut _, (n * mem::size_of::<f32>()) as u64, MTLResourceOptions::StorageModeShared);

    // Load Metal shader
    let source = include_str!("add_vectors.metal");
    let library = device.new_library_with_source(source, &CompileOptions::new()).unwrap();
    let kernel = library.get_function("add_vectors", None).unwrap();
    let pipeline_state = device.new_compute_pipeline_state_with_function(&kernel).unwrap();

    // Create command buffer and encoder
    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&a_buffer), 0);
    encoder.set_buffer(1, Some(&b_buffer), 0);
    encoder.set_buffer(2, Some(&c_buffer), 0);
    // Pass n as bytes directly (more efficient than a buffer)
    let n_u32 = n as u32;
    encoder.set_bytes(3, mem::size_of::<u32>() as u64, &n_u32 as *const _ as *const _);

    // Dispatch threads
    let threads_per_group = MTLSize::new(256, 1, 1);
    let num_threadgroups = MTLSize::new((n as u64 + 255) / 256, 1, 1);
    encoder.dispatch_thread_groups(num_threadgroups, threads_per_group);
    encoder.end_encoding();

    // Commit command buffer
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read results
    let result_ptr = c_buffer.contents() as *const f32;
    let result = unsafe { std::slice::from_raw_parts(result_ptr, n) };
    println!("{:?}", result);
}
