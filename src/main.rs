use image::{imageops::FilterType, ImageBuffer, Luma, Pixel};
use ort::{
    environment::Environment, tensor::OrtOwnedTensor, value::Value, LoggingLevel,
    NdArrayExtensions, OrtResult, SessionBuilder,
};

const IMAGE_BYTES: &[u8] = include_bytes!("mnist_5.jpg");
const MODEL_BYTES: &[u8] = include_bytes!("mnist-8.onnx");

fn main() -> OrtResult<()> {
    let environment = Environment::builder()
        .with_name("ort + WASM")
        .with_log_level(LoggingLevel::Warning)
        .build()?
        .into_arc();

    let session = SessionBuilder::new(&environment)?
        .with_model_from_memory(MODEL_BYTES)
        .expect("Could not load model");

    let input0_shape: Vec<usize> = session.inputs[0].dimensions().map(|d| d.unwrap()).collect();

    // Load image and resize to model's shape, converting to RGB format
    let image_buffer: ImageBuffer<Luma<u8>, Vec<u8>> = image::load_from_memory(IMAGE_BYTES)
        .unwrap()
        .resize(
            input0_shape[2] as u32,
            input0_shape[3] as u32,
            FilterType::Nearest,
        )
        .to_luma8();

    let array = ndarray::CowArray::from(
        ndarray::Array::from_shape_fn((1, 1, 28, 28), |(_, c, j, i)| {
            let pixel = image_buffer.get_pixel(i as u32, j as u32);
            let channels = pixel.channels();

            // range [0, 255] -> range [0, 1]
            (channels[c] as f32) / 255.0
        })
        .into_dyn(),
    );

    // Batch of 1
    let input_tensor_values = vec![Value::from_array(session.allocator(), &array)?];

    // Perform the inference
    let outputs: Vec<Value> = session.run(input_tensor_values)?;

    let output: OrtOwnedTensor<_, _> = outputs[0].try_extract()?;
    let mut probabilities: Vec<(usize, f32)> = output
        .view()
        .softmax(ndarray::Axis(1))
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();

    // Sort probabilities so highest is at beginning of vector.
    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("{probabilities:?}");

    Ok(())
}
