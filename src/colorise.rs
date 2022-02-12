//! ## RIMCOL - Rust Image Colorizer
//! A rust crate for converting a black and white grayscale
//! image to an rgb image. This makes use of a CNN model written
//! in tensorflow. This only gives a probable image colorization
//! and is being constantly evolved to give higher accuracy.

extern crate lab;
extern crate tensorflow;
use image::io::Reader as ImageReader;
use image::DynamicImage;
use image::GenericImageView;
use image::{image_dimensions, save_buffer_with_format};
use lab::Lab;
use lab::{labs_to_rgbs, rgbs_to_labs};
use std::error::Error;
use std::fmt;
use std::result::Result;
use tensorflow::{
    Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor,
    DEFAULT_SERVING_SIGNATURE_DEF_KEY,
};

#[derive(Debug)]
struct CustomError(String);

impl Error for CustomError {}

impl fmt::Display for CustomError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Error: {}", self.0)
    }
}

/// The main driver function to conevert from grayscale to RGB.
///
/// Takes 2 inputs: `path_to_input` specifying the source file location
/// and `path_for_output` to determine the location at which the colorised
/// rgb image should be saved.
pub fn bw_to_rgb(path_to_input: String, path_for_output: String) -> Result<(), Box<dyn Error>> {
    let img_dimensions: (u32, u32) = image_dimensions(&path_to_input)?;
    if img_dimensions != (256, 256) {
        return Err(Box::new(CustomError(format!(
            "The dimensions of the image ({}, {}) are not supported, should be (256, 256)",
            img_dimensions.0, img_dimensions.1
        ))));
    }

    let model_dir: String = format!("/home/{}/beta", whoami::username());

    let input_image: DynamicImage = ImageReader::open(&path_to_input)?.decode()?;
    let mut input_rgbs: Vec<[u8; 3]> = Vec::new();
    for (_, (_, _, pixel)) in input_image.pixels().enumerate() {
        let rgb: [u8; 3] = [pixel.0[0], pixel.0[1], pixel.0[2]];
        input_rgbs.push(rgb)
    }
    let input_labs: Vec<Lab> = rgbs_to_labs(&input_rgbs);

    let mut input_layer: Tensor<f32> = Tensor::new(&[1, 256, 256, 1]);

    let mut i = 0;

    for pixel in &input_labs {
        input_layer[i] = pixel.l as f32;
        i += 1;
    }

    let mut graph = Graph::new();
    let bundle = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, model_dir)?;
    let session = &bundle.session;

    let signature = bundle
        .meta_graph_def()
        .get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)?;
    let x_info = signature.get_input("input_1")?;
    let op_x = &graph.operation_by_name_required(&x_info.name().name)?;
    let output_info = signature.get_output("up_sampling2d_2")?;
    let op_output = &graph.operation_by_name_required(&output_info.name().name)?;

    let mut session_args = SessionRunArgs::new();
    session_args.add_feed(op_x, 0, &input_layer);
    let token_output = session_args.request_fetch(op_output, 0);
    session.run(&mut session_args)?;

    let output: Tensor<f32> = session_args.fetch(token_output)?;

    let mut output_labs: Vec<Lab> = Vec::new();

    for i in 0..256 {
        for j in 0..256 {
            output_labs.push(Lab {
                l: input_labs[i * 256 + j].l,
                a: (output.get(&[0, i as u64, j as u64, 0]) * 128f32),
                b: (output.get(&[0, i as u64, j as u64, 1]) * 128f32),
            });
        }
    }

    let output_rgb = labs_to_rgbs(&output_labs);
    let mut output_buffer: Vec<u8> = Vec::new();

    for pixel in output_rgb {
        output_buffer.push(pixel[0]);
        output_buffer.push(pixel[1]);
        output_buffer.push(pixel[2]);
    }
    save_buffer_with_format(
        path_for_output,
        &output_buffer,
        256,
        256,
        image::ColorType::Rgb8,
        image::ImageFormat::Png,
    )
    .unwrap();

    Ok(())
}
