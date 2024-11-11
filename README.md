
# embedded-huffman

A paginated, streaming library for Huffman coding designed for embedded systems. This library provides efficient compression and decompression with minimal memory overhead, making it suitable for resource-constrained environments.

## Features

- No-std compatible, (`alloc` required)
- Streaming compression/decompression
- Paginated output for NAND flash pages
- Adaptive Huffman coding that rebuilds frequency tables periodically
- Zero-copy design with minimal allocations
- Optional CLI tool for file compression

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
embedded-huffman = "0.1.0"
```

Include `extern crate alloc;` in your crate root.

Create an implementation for:

```rust
/// A function that takes a reference to the page and writes it to NAND
pub type WritePageFutureFn<E> =
    Box<dyn for<'a> Fn(&'a [u8]) -> Pin<Box<dyn Future<Output = Result<(), E>> + 'a>>>;

/// A function that takes a mutable reference to the page and fills it with bytes from NAND
/// The future returns true if there are more pages that could be read
pub type ReadPageFutureFn<E> =
    Box<dyn for<'a> Fn(&'a mut [u8]) -> Pin<Box<dyn Future<Output = Result<bool, E>> + 'a>>>;
```

## Usage

### Command Line Interface

The CLI tool provides simple compression and decompression capabilities:

```bash
# Compress a file
hfz < input.txt > compressed.hfz

# Decompress a file
hfz -d < compressed.hfz > decompressed.txt

# Customize page size and threshold
hfz -s 8192 -t 512 < input.txt > compressed.hfz
```

### Library Usage

Basic compression example:

```rust
use embedded_huffman::{Encoder, BufferedPageWriter};

const PAGE_SIZE: usize = 2048;
const PAGE_THRESHOLD: usize = 4;

// Create encoder and writer
let mut encoder = Encoder::new(PAGE_SIZE, PAGE_THRESHOLD);
let mut writer = BufferedPageWriter::new(PAGE_SIZE, flush_fn);

// Compress data
for byte in data {
    encoder.sink(byte, &mut writer).await?;
}
encoder.flush(&mut writer).await?;
```

Basic decompression example:

```rust
use embedded_huffman::{Decoder, BufferedPageReader};

// Create decoder and reader
let mut decoder = Decoder::new(PAGE_SIZE, PAGE_THRESHOLD);
let mut reader = BufferedPageReader::new(PAGE_SIZE, read_fn);

// Decompress data
while let Some(byte) = decoder.drain(&mut reader).await? {
    // Process decompressed byte
}
```

## How It Works

The library implements a streaming Huffman coding algorithm with the following key features:

1. **Paginated Output**: Data is written in fixed-size pages suitable for NAND flash storage.

2. **Adaptive Tables**: The Huffman tree is rebuilt periodically based on symbol frequencies in the previous N pages.

3. **Memory Efficient**: Uses a minimal memory footprint with most operations being zero-copy.

4. **Streaming Interface**: Data can be processed incrementally without loading everything into memory.

## Performance

The library includes benchmarks for compression and decompression performance. Run them with:

```bash
cargo bench --all-features
```

## Testing

The codebase includes:

- Unit tests
- Fuzzing tests (in the `fuzz` directory)
- End-to-end roundtrip tests
- Property-based tests

Run the test suite with:

```bash
cargo test
```

### Fuzzing

Fuzzing ran for 4 days on Mac Studio `cargo +nightly fuzz run fuzz_target_1 -- -max_len=10000000 -jobs=20`

The fuzz test included a roundtrip of bytes through the encoder and decoder.

## CLI Installation

```bash
# Install
cargo install --path . --features std,cli
```

# Example

In this example with pipe viewer, you can see the compression and decompression rates:
```bash
jacobtrueb@Jacobs-Mac-Studio embedded-huffman % hfz < random.bin | pv > random.bin.hfz   
1.00GiB 0:00:10 [96.6MiB/s] [           <=>           ]
jacobtrueb@Jacobs-Mac-Studio embedded-huffman % hfz -d < random.bin.hfz | pv > random.bin.unhfz 
1.00GiB 0:00:18 [56.4MiB/s] [    <=>                  ]
```

The page_threshold is approached with exponential backoff, so it may take a while to have pages spaced out as specified. To mitigate issues with non-representative frequency tables in the early data. The first Huffman table is built after 1 page, then after 2 pages, then 4, 8, 16, etc. until the threshold is reached.

```rust
///!
///! This is a simple CLI that reads bytes from stdin and writes Huffman-compressed data to stdout.
///!
///! The -d flag indicates if decompressing bytes emitted by this program.
///! The -s flag specifies the page size (must be power of 2).
///! The -t flag specifies the page threshold for rebuilding the Huffman table.
///!
```
