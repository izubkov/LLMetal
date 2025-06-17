use objc2_metal::*;
use objc2_foundation::NSString;
use std::mem;
use std::ptr::NonNull;
use std::ffi::c_void;

fn main() {
    // Initialize Metal device and command queue
    let device = MTLCreateSystemDefaultDevice().expect("No Metal device found");
    let command_queue = device.newCommandQueue().expect("Failed to create command queue");

    println!("{:?}", device);

    // Create buffers
    let n = 1024;
    let a = vec![1.0f32; n];
    let b = vec![2.0f32; n];
    let mut c = vec![0.0f32; n];

    let a_id = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::from(&a[0]).cast::<c_void>(),
            n * mem::size_of::<f32>(),
            MTLResourceOptions::StorageModeShared,
        )
    };
    let a_buffer = a_id.as_deref();
    let b_id = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::from(&b[0]).cast::<c_void>(),
            n * mem::size_of::<f32>(),
            MTLResourceOptions::StorageModeShared,
        )
    };
    let b_buffer = b_id.as_deref();
    let c_id = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::from(&mut c[0]).cast::<c_void>(),
            n * mem::size_of::<f32>(),
            MTLResourceOptions::StorageModeShared,
        )
    };
    let c_buffer = c_id.as_deref();

    // Load Metal shader
    let source: &'static str = include_str!("add_vectors.metal");
    let source_nsstring = NSString::from_str(source);
    let library = device.newLibraryWithSource_options_error(&source_nsstring, None).unwrap();
    let function_name = NSString::from_str("add_vectors");
    let kernel = library.newFunctionWithName(&function_name).unwrap();
    let pipeline_state = device.newComputePipelineStateWithFunction_error(&kernel).unwrap();

    // Create command buffer and encoder
    let command_buffer = command_queue.commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create encoder");
    unsafe {
        encoder.setComputePipelineState(&pipeline_state);
        encoder.setBuffer_offset_atIndex(a_buffer, 0, 0);
        encoder.setBuffer_offset_atIndex(b_buffer, 0, 1);
        encoder.setBuffer_offset_atIndex(c_buffer, 0, 2);
    }
    
    // Pass n as bytes directly (more efficient than a buffer)
    let n_u32 = n as u32;
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::from(&n_u32).cast::<c_void>(),
            mem::size_of::<u32>(),
            3,
        );
    }

    // Dispatch threads
    let threads_per_group = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let num_threadgroups = MTLSize {
        width: (n + 255) / 256,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(num_threadgroups, threads_per_group);
    encoder.endEncoding();

    // Commit command buffer
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    // Read results
    let result_buffer = c_buffer.expect("Failed to create MTLBuffer");
    let result_ptr = result_buffer.contents().cast::<f32>().as_ptr();
    let result = unsafe { std::slice::from_raw_parts(result_ptr, n) };
    println!("{:?}", result);
}
