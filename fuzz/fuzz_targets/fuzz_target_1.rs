#![no_main]

use embedded_huffman::*;
use libfuzzer_sys::fuzz_target;
use std::cell::RefCell;
use std::rc::Rc;

fuzz_target!(|data: &[u8]| {
    const PAGE_SIZE: usize = 2048;
    const PAGE_THRESHOLD: usize = 1024;

    let buf: Vec<u8> = Vec::new();
    let buf = Rc::new(RefCell::new(buf));
    let wtr_buf = buf.clone();
    let rdr_buf = buf.clone();

    // Encoding
    let flush_page: FlushFutureFn<()> = Box::new(move |page| {
        let buf = wtr_buf.clone();
        Box::pin(async move {
            let mut buf = buf.borrow_mut();
            buf.extend_from_slice(page);
            Ok(())
        })
    });
    let mut wtr = BufferedPageWriter::new(PAGE_SIZE, flush_page);
    let mut encoder = Encoder::new(PAGE_SIZE, PAGE_THRESHOLD);

    // Decoding
    let read_page: ReadPageFutureFn<()> = Box::new(move |page| {
        let buf = rdr_buf.clone();
        Box::pin(async move {
            let mut buf = buf.borrow_mut();
            assert!(buf.len() % PAGE_SIZE == 0);
            if buf.is_empty() {
                page.fill(0xFF);
            } else {
                let drained = buf.drain(..PAGE_SIZE);
                page[..drained.len()]
                    .iter_mut()
                    .zip(drained)
                    .for_each(|(p, b)| *p = b);
            }
            Ok(())
        })
    });
    let mut rdr = BufferedPageReader::new(PAGE_SIZE, read_page);
    let mut decoder = Decoder::new(PAGE_SIZE, PAGE_THRESHOLD);

    // Roundtrip
    smol::block_on(async {
        // Put in
        for value in data {
            encoder.sink(*value, &mut wtr).await.unwrap();
        }
        encoder.flush(&mut wtr).await.unwrap();

        // Take out
        let mut e2e = Vec::new();
        while let Some(byte) = decoder.drain(&mut rdr).await.unwrap() {
            e2e.push(byte);
        }

        // Check
        assert_eq!(e2e, data);
    });
});
