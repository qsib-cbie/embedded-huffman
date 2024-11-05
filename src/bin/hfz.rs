///!
///! This is a simple CLI that reads bytes from stdin and writes Huffman-compressed data to stdout.
///!
///! The -d flag indicates if decompressing bytes emitted by this program.
///! The -s flag specifies the page size (must be power of 2).
///! The -t flag specifies the page threshold for rebuilding the Huffman table.
///!
extern crate alloc;

#[cfg(feature = "cli")]
extern crate smol;

use embedded_huffman::{
    BufferedPageReader, BufferedPageWriter, Decoder, Encoder, FlushFutureFn, ReadPageFutureFn,
};
use std::io::{self, Read, Write};
use std::process;

/// 16KiB is the default page size for the CLI
const DEFAULT_PAGE_SIZE: usize = 16384;
/// 256MiB is the default huffman tree rebuilding threshold for the CLI (when page size is 16KiB)
const DEFAULT_PAGE_THRESHOLD: usize = 16384;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut decompress = false;
    let mut page_size = DEFAULT_PAGE_SIZE;
    let mut page_threshold = DEFAULT_PAGE_THRESHOLD;
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "-d" => {
                decompress = true;
            }
            "-s" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Missing argument for -s");
                    process::exit(1);
                }
                match args[i].parse::<usize>() {
                    Ok(s) => {
                        if !s.is_power_of_two() {
                            eprintln!("Page size must be a power of 2");
                            process::exit(1);
                        }
                        page_size = s;
                    }
                    Err(_) => {
                        eprintln!("Invalid page size");
                        process::exit(1);
                    }
                }
            }
            "-t" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Missing argument for -t");
                    process::exit(1);
                }
                match args[i].parse::<usize>() {
                    Ok(t) => {
                        page_threshold = t;
                    }
                    Err(_) => {
                        eprintln!("Invalid page threshold");
                        process::exit(1);
                    }
                }
            }
            _ => {
                eprintln!("Usage: {} [-d] [-s page_size] [-t page_threshold]", args[0]);
                eprintln!("  -d: decompress mode");
                eprintln!("  -s: page size (must be power of 2)");
                eprintln!("  -t: page threshold for rebuilding Huffman table");
                process::exit(1);
            }
        }
        i += 1;
    }

    if decompress {
        let read_page: ReadPageFutureFn<io::Error> = Box::new(move |page| {
            Box::pin(async move {
                let stdin = io::stdin();
                let mut stdin = stdin.lock();
                let mut bytes_read = 0;
                while bytes_read < page.len() {
                    match stdin.read(&mut page[bytes_read..]) {
                        Ok(0) => {
                            // EOF reached, fill rest with 0xFF
                            page[bytes_read..].fill(0xFF);
                            break;
                        }
                        Ok(n) => bytes_read += n,
                        Err(e) => return Err(e),
                    }
                }
                Ok(())
            })
        });

        let mut rdr = BufferedPageReader::new(page_size, read_page);
        let mut decoder = Decoder::new(page_size, page_threshold);

        smol::block_on(async {
            let stdout = io::stdout();
            let mut stdout = stdout.lock();
            while let Some(byte) = decoder.drain(&mut rdr).await.unwrap() {
                stdout.write_all(&[byte]).unwrap();
            }
        });
    } else {
        let flush_page: FlushFutureFn<io::Error> = Box::new(move |page| {
            Box::pin(async move {
                let stdout = io::stdout();
                let mut stdout = stdout.lock();
                stdout.write_all(page).map_err(|e| e)
            })
        });

        let mut wtr = BufferedPageWriter::new(page_size, flush_page);
        let mut encoder = Encoder::new(page_size, page_threshold);

        smol::block_on(async {
            let mut buf = vec![0u8; page_size];
            let stdin = io::stdin();
            let mut stdin = stdin.lock();
            loop {
                match stdin.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        for byte in &buf[..n] {
                            encoder.sink(*byte, &mut wtr).await.unwrap();
                        }
                    }
                    Err(e) => {
                        eprintln!("Error reading stdin: {}", e);
                        process::exit(1);
                    }
                }
            }
            encoder.flush(&mut wtr).await.unwrap();
        });
    }
}
