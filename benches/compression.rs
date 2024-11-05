use divan::Bencher;
use embedded_huffman::*;
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    divan::main();
}

#[divan::bench]
fn compress(bencher: Bencher) {
    // Generate test data - 1MB of pseudo-random bytes
    let test_data: Vec<u8> = (0..1024 * 1024).map(|i| ((i * 31) % 256) as u8).collect();

    // Set up shared buffer and readers/writers
    let buf: Vec<u8> = Vec::new();
    let buf = Rc::new(RefCell::new(buf));
    let wtr_buf = buf.clone();

    let flush_page: WritePageFutureFn<()> = Box::new(move |page| {
        let buf = wtr_buf.clone();
        Box::pin(async move {
            let mut buf = buf.borrow_mut();
            buf.extend_from_slice(page);
            Ok(())
        })
    });

    const PAGE_SIZE: usize = 2048;
    const PAGE_THRESHOLD: usize = 1024;
    let mut wtr = BufferedPageWriter::new(PAGE_SIZE, flush_page);
    let mut encoder = Encoder::new(PAGE_SIZE, PAGE_THRESHOLD);

    bencher.bench_local(move || {
        // Reset state
        buf.borrow_mut().clear();
        encoder.reset();
        wtr.reset();

        // Compress
        smol::block_on(async {
            for &value in &test_data {
                encoder.sink(value, &mut wtr).await.unwrap();
            }
            encoder.flush(&mut wtr).await.unwrap();
        });
    });
}

#[divan::bench]
fn decompress(bencher: Bencher) {
    // Generate test data - 1MB of pseudo-random bytes
    let test_data: Vec<u8> = (0..1024 * 1024).map(|i| ((i * 31) % 256) as u8).collect();

    // Set up shared buffer and readers/writers
    let buf: Vec<u8> = Vec::new();
    let buf = Rc::new(RefCell::new(buf));
    let wtr_buf = buf.clone();
    let rdr_buf = buf.clone();

    let flush_page: WritePageFutureFn<()> = Box::new(move |page| {
        let buf = wtr_buf.clone();
        Box::pin(async move {
            let mut buf = buf.borrow_mut();
            buf.extend_from_slice(page);
            Ok(())
        })
    });

    const PAGE_SIZE: usize = 2048;
    const PAGE_THRESHOLD: usize = 1024;
    let mut wtr = BufferedPageWriter::new(PAGE_SIZE, flush_page);
    let mut encoder = Encoder::new(PAGE_SIZE, PAGE_THRESHOLD);

    let read_page: ReadPageFutureFn<()> = Box::new(move |page| {
        let buf = rdr_buf.clone();
        Box::pin(async move {
            let mut buf = buf.borrow_mut();
            assert!(buf.len() % PAGE_SIZE == 0);
            if buf.is_empty() {
                page.fill(0xFF);
                Ok(false)
            } else {
                let drained = buf.drain(..PAGE_SIZE);
                page[..drained.len()]
                    .iter_mut()
                    .zip(drained)
                    .for_each(|(p, b)| *p = b);
                Ok(true)
            }
        })
    });

    let mut rdr = BufferedPageReader::new(PAGE_SIZE, read_page);
    let mut decoder = Decoder::new(PAGE_SIZE, PAGE_THRESHOLD);

    bencher.bench_local(move || {
        // Reset state
        buf.borrow_mut().clear();
        encoder.reset();
        wtr.reset();
        decoder.reset();
        rdr.reset();

        // Compress
        smol::block_on(async {
            for &value in &test_data {
                encoder.sink(value, &mut wtr).await.unwrap();
            }
            encoder.flush(&mut wtr).await.unwrap();
        });

        // Decompress and verify
        smol::block_on(async {
            let mut idx = 0;
            while let Some(byte) = decoder.drain(&mut rdr).await.unwrap() {
                assert_eq!(byte, test_data[idx]);
                idx += 1;
            }
            assert_eq!(idx, test_data.len());
        });
    });
}
