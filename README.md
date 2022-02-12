<div align="center">

# **rimcol**

**rimcol - Rust Image Coloriser** is a command line tool for converting grayscale images to RGB images.

</div>

---

**rimcol** stands for *rust image coloriser*. It is a command line tool written in rust to predict the rgb color pallete of a grayscale image using a CNN model written in tensorflow and python. The input image should be a **256*256** grayscale image and the output is an RGB image of the same dimension.

**Note**: The model is still in its early stages and is evolved to be more accurate with each iteration.

---

<h1>Command-line options</h1>

* **-i**, **--path-to-input** (Parameter): Pass the input grayscale image location.
* **-o**, **--path-for-output** (Parameter): Pass the output location for saving the RGB image.
* **-h**, **--help**: Get the help info.

---

<h1>Installation</h1>

rimcol currently can be used in 3 ways, install via [cargo](https://doc.rust-lang.org/cargo/), build from source and use as a rust package.

Before installing, download the model from <a href="https://drive.google.com/drive/folders/1-Q--MxdkfbO26UHIKBlrkRSY9TT2VFG4?usp=sharing" target="_blank">here</a>,
extract it and place the folder at `/home/<user>/`.

**Note**: rimcol depends on tensorflow for running the CNN model, so you should have tensorflow prerequisites installed. Refer [tensorflow-rust](https://github.com/tensorflow/rust).

## Manual installation from GitHub

Compiled binary versions of rimcol are uploaded to GitHub when a release is made.
You can install rimcol manually by [downloading a release](https://github.com/ligmitz/rimcol/releases), extracting it, and copying the binary to a directory in your `$PATH`, such as `/usr/local/bin`.

## Cargo

If you already have a Rust environment set up, you can use the `cargo install` command:

    $ cargo install rimcol

Cargo will build the `rimcol` binary and place it in `$HOME/.cargo`.

## As a package

If you want to use rimcol as a package in your rust project, add this to your `Cargo.toml` file:

```
rimcol = "0.1.0"
```

---

<a id="development">
<h1>Development

<a href="https://blog.rust-lang.org/2021/12/02/Rust-1.57.0.html">
    <img src="https://img.shields.io/badge/rustc-1.57.0+-lightgray.svg" alt="Rust 1.57.0+" />
</a>

<a href="https://github.com/ogham/exa/blob/master/LICENCE">
    <img src="https://img.shields.io/badge/licence-MIT-green" alt="MIT Licence" />
</a>
</h1></a>

rimcol is written in [Rust](https://www.rust-lang.org/).
You will need rustc version 1.57.0 or higher.
The recommended way to install Rust for development is from the [official download page](https://www.rust-lang.org/tools/install), using rustup.

Once Rust is installed, you can compile the rimcol directory cloned from github with Cargo:

    $ cargo build

- If you want to compile a version for yourself, run `cargo build --release`.

Copy the resulting binary, which will be in the `target/release` directory, into a folder in your `$PATH` such as `/usr/local/bin`.
