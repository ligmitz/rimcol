use clap::Parser;
use rimcol::bw_to_rgb;
use std::error::Error;
use std::result::Result;

/// Convert a black and white (grayscale) image to colored (RGB) image
#[derive(Parser)]
struct Cli {
    /// Path of input image
    #[clap(short = 'i', long)]
    path_to_input: String,

    /// Path for saving output image
    #[clap(short = 'o', long)]
    path_for_output: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Cli = Cli::parse();
    bw_to_rgb(args.path_to_input, args.path_for_output)?;
    Ok(())
}
