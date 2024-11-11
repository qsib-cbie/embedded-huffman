#![no_std]
extern crate alloc;
#[cfg(any(feature = "std", test))]
extern crate std;

///!
///! This is a library for streaming sensor data through
///! an Encoder that writes its frequency table and huffman coding
///! to NAND pages on a sensor. The Encoder includes some
///! metadata to allow flushing bytes directly to NAND, including
///! page headers every N bytes to indicate bits encoded in the page.
///! The Decoder knows how to read these pages, inflate the huffman
///! coding from the frequency table, and decode the sensor data.
///!
///! The Encoder and Decoder use the previous 1MiB of frequencies
///! to build the frequency table used for the next 1MiB of data.
///!
///! This allows the sensor data to change over time, and the
///! encoder and decoder to adapt to changes int the sensor data.
///!
use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;
use core::ptr::copy_nonoverlapping;

mod tree;
use tree::*;

/// The Huffman code is built from bytes, so the symbol count is 2^8
const WORD_SIZE: usize = 8;
const SYMBOL_COUNT: usize = 1 << WORD_SIZE;

// Predefined bit shifts to remove some of the bit shifting
static PRESHIFTED7: [u8; 2] = [0b0000_0000, 0b1000_0000];
static PRESHIFTED6: [u8; 2] = [0b0000_0000, 0b0100_0000];
static PRESHIFTED5: [u8; 2] = [0b0000_0000, 0b0010_0000];
static PRESHIFTED4: [u8; 2] = [0b0000_0000, 0b0001_0000];
static PRESHIFTED3: [u8; 2] = [0b0000_0000, 0b0000_1000];
static PRESHIFTED2: [u8; 2] = [0b0000_0000, 0b0000_0100];
static PRESHIFTED1: [u8; 2] = [0b0000_0000, 0b0000_0010];
static PRESHIFTED0: [u8; 2] = [0b0000_0000, 0b0000_0001];

/// A huffman encoder that writes successive tables and data to pages
pub struct Encoder {
    page_size: usize,
    page_threshold: usize,
    page_threshold_limit: usize,
    page_count: usize,
    state: EncodeState,
    word_batch: Vec<u8>,
    weights: [u32; SYMBOL_COUNT],
    code_table: Vec<CodeEntry>,
    visit_deque: VecDeque<(Node, Vec<usize>)>,
    done: bool,
    #[cfg(feature = "ratio")]
    bytes_in: usize,
    #[cfg(feature = "ratio")]
    bytes_out: usize,
}

/// The state while sinking bytes into the encoder
#[derive(Debug, Copy, Clone)]
enum EncodeState {
    /// The encoder is gathering enough data for the initial table
    Init,
    /// The encoder is writing the frequency table to NAND.
    /// The huffman tree is built from the frequencies.
    Table,
    /// The encoder is writing encoded data to NAND.
    /// The data is encoded with a table derived from the previous huffman coding.
    Data,
    /// The encoder is in an error state. This is terminal.
    Error,
}

impl Encoder {
    /// Create an encoder that writes into pages of `page_size` bytes
    /// and rebuilds the frequency table every `page_threshold` pages.
    pub fn new(page_size: usize, page_threshold: usize) -> Encoder {
        Encoder {
            page_size,
            page_threshold: 1, // exponential backoff until page_threshold_limit
            page_threshold_limit: page_threshold,
            page_count: 0,
            state: EncodeState::Init,
            word_batch: Vec::with_capacity(page_size),
            weights: [1; SYMBOL_COUNT], // always assume each symbol is present at least once
            code_table: Vec::with_capacity(SYMBOL_COUNT),
            visit_deque: VecDeque::with_capacity(SYMBOL_COUNT * 2 - 1),
            done: false,
            #[cfg(feature = "ratio")]
            bytes_in: 0,
            #[cfg(feature = "ratio")]
            bytes_out: 0,
        }
    }

    /// Prepare the encoder for a new round of encoding
    /// this keeps allocations, so its cheaper than a Encoder::new call
    pub fn reset(&mut self) {
        self.page_count = 0;
        self.page_threshold = 1;
        self.state = EncodeState::Init;
        self.word_batch.clear();
        self.weights.fill(1);
        self.code_table.clear();
        self.done = false;
        #[cfg(feature = "ratio")]
        {
            self.bytes_in = 0;
            self.bytes_out = 0;
        }
    }

    /// Get the ratio of bytes in to bytes out
    #[cfg(feature = "ratio")]
    pub fn ratio(&self) -> f32 {
        self.bytes_in as f32 / self.bytes_out as f32
    }

    #[inline(always)]
    /// Check if a batch of bytes fits in the current batch
    pub fn batch_fits(&self, bytes: usize) -> bool {
        self.word_batch.len() + bytes < self.page_size
    }

    #[inline(always)]
    /// Sink a batch of bytes without checking if the batch fits
    pub unsafe fn batch_sink(&mut self, bytes: &[u8]) {
        // Optimized for Cortex-M4 where there is no branch prediction
        #[cfg(feature = "ratio")]
        {
            self.bytes_in += bytes.len();
        }
        // SAFETY:
        // we know weights is 256 elements long and is indexed by a u8 [0..256]
        // we know word_batch is < len because we check it below
        // we can't use get_unchecked_mut because we have uninitialized memory
        let weights_ptr = self.weights.as_mut_ptr();
        let word_batch_ptr = self.word_batch.as_mut_ptr();
        for byte in bytes {
            *weights_ptr.add(*byte as usize) += 1;
        }
        let idx = self.word_batch.len();
        let dst = word_batch_ptr.add(idx);
        let src = bytes.as_ptr();
        copy_nonoverlapping(src, dst, bytes.len());
        self.word_batch.set_len(idx + bytes.len());
    }

    /// Put a byte into the encoder
    #[inline(always)]
    pub async fn sink<E>(&mut self, byte: u8, output: &mut impl PageWriter<E>) -> Result<bool, E> {
        self.crank(output, Some(byte)).await
    }

    /// Finish encoding and flush any remaining bytes
    pub async fn flush<E>(&mut self, output: &mut impl PageWriter<E>) -> Result<bool, E> {
        self.crank(output, None).await
    }

    #[inline(always)]
    async fn crank<E>(
        &mut self,
        output: &mut impl PageWriter<E>,
        byte: Option<u8>,
    ) -> Result<bool, E> {
        let finish = if let Some(byte) = byte {
            // Optimized for Cortex-M4 where there is no branch prediction
            #[cfg(feature = "ratio")]
            {
                self.bytes_in += 1;
            }
            // SAFETY:
            // we know weights is 256 elements long and is indexed by a u8 [0..256]
            // we know word_batch is < len because we check it below
            // we can't use get_unchecked_mut because we have uninitialized memory
            unsafe {
                let weights_ptr = self.weights.as_mut_ptr();
                let word_batch_ptr = self.word_batch.as_mut_ptr();
                *weights_ptr.add(byte as usize) += 1;
                let idx = self.word_batch.len();
                *word_batch_ptr.add(idx) = byte;
                self.word_batch.set_len(idx + 1);
            }

            // Hot path for encoding data
            if matches!(self.state, EncodeState::Data) & (self.word_batch.len() < self.page_size) {
                return Ok(self.done);
            }

            false
        } else {
            true
        };

        loop {
            match self.state {
                EncodeState::Error => {
                    unreachable!()
                }
                EncodeState::Init => {
                    if self.word_batch.len() == self.page_size || finish {
                        self.state = EncodeState::Table;
                        output.reset();
                    } else {
                        return Ok(self.done);
                    }
                }
                EncodeState::Table => {
                    // The table is SYMBOL_COUNT * 4 bytes long, [0 -> count, 1 -> count, 2 -> count, ...]
                    for &weight in &self.weights {
                        output.write_u32le(weight as u32);
                    }

                    // Write out bytes
                    #[cfg(feature = "ratio")]
                    {
                        self.bytes_out += self.page_size;
                    }
                    match output.flush().await {
                        Ok(done) => {
                            self.done |= done;
                        }
                        Err(e) => {
                            self.state = EncodeState::Error;
                            return Err(e);
                        }
                    }

                    // Build the Huffman tree from the symbol frequencies
                    let root = build_tree(&self.weights);

                    // Create the bit representation of the tree
                    self.code_table.clear();
                    self.code_table.resize(SYMBOL_COUNT, Default::default());
                    build_code_table(root, &mut self.code_table, &mut self.visit_deque);

                    // Always assume each symbol is present at least once
                    self.weights.fill(1);

                    // Ready to start encoding data
                    self.state = EncodeState::Data;
                }
                EncodeState::Data => {
                    // batching is only done for performance reasons ... hypothetically
                    if (self.word_batch.len() < self.page_size) & !finish {
                        return Ok(self.done);
                    }

                    // write all of the bits in the current batch using the current code table
                    let mut drain = None;
                    for (idx, byte) in self.word_batch.iter().enumerate() {
                        let code = unsafe { self.code_table.get_unchecked(*byte as usize) };

                        // If the word does not fit in the current page, then we need to advance pages
                        if !output.write_code(code) {
                            drain = Some(idx);
                            break;
                        }
                    }

                    if let Some(drain) = drain {
                        // Remove words that are emitted
                        self.word_batch.drain(..drain);

                        // Move to the next page, writing out this page
                        #[cfg(feature = "ratio")]
                        {
                            self.bytes_out += self.page_size;
                        }
                        match output.flush().await {
                            Ok(done) => {
                                self.done |= done;
                            }
                            Err(e) => {
                                self.state = EncodeState::Error;
                                return Err(e);
                            }
                        }

                        // Update the page count and check if we need to rebuild the table
                        self.page_count += 1;
                        if self.page_count > self.page_threshold {
                            self.page_count = 0;
                            self.page_threshold =
                                self.page_threshold_limit.min(self.page_threshold * 2);
                            self.state = EncodeState::Table;
                        } else {
                            self.state = EncodeState::Data;
                        }
                    } else {
                        self.word_batch.clear();

                        // We are done if all the words sunk, emit the final page
                        if finish {
                            #[cfg(feature = "ratio")]
                            {
                                self.bytes_out += self.page_size;
                            }
                            return output.flush().await.and_then(|done| {
                                self.done |= done;
                                Ok(self.done)
                            });
                        }
                    }
                }
            }
        }
    }
}

#[allow(async_fn_in_trait)]
pub trait PageWriter<E> {
    async fn flush(&mut self) -> Result<bool, E>;

    fn position(&self) -> usize;
    fn reset(&mut self);
    fn write_header(&mut self, header: u32);
    fn write_u32le(&mut self, value: u32);
    fn write_code(&mut self, code: &CodeEntry) -> bool;
}

pub struct BufferedPageWriter<E> {
    /// Position in the current page in bits
    bits_written: usize,
    /// Pre-calculated page size in bits
    page_size: usize,
    /// Vec<bool> pre-cast to u8, representing bits that need to be chunked into bytes
    bits: Vec<usize>,
    /// A page of bytes that need to be flushed to NAND
    bytes: Vec<u8>,
    /// A function that takes a ref to the page and writes it to NAND
    flush_page: WritePageFutureFn<E>,
    /// Done
    done: bool,
}

/// A function that takes a reference to the page and writes it to NAND
pub type WritePageFutureFn<E> =
    Box<dyn for<'a> Fn(&'a [u8]) -> Pin<Box<dyn Future<Output = Result<bool, E>> + 'a>>>;

impl<E> BufferedPageWriter<E> {
    pub fn new(page_size: usize, flush: WritePageFutureFn<E>) -> BufferedPageWriter<E> {
        // Allocate a buffer for the page
        let mut buf: Vec<u8> = Vec::with_capacity(page_size);
        unsafe { buf.set_len(page_size) };
        BufferedPageWriter {
            page_size: 8 * page_size,
            bits: Vec::with_capacity(8 * 2),
            bytes: buf,
            bits_written: 0,
            flush_page: flush,
            done: false,
        }
    }
}

impl<E> PageWriter<E> for BufferedPageWriter<E> {
    /// current bit position in the page
    #[inline(always)]
    fn position(&self) -> usize {
        self.bits_written
    }

    /// reset the page
    fn reset(&mut self) {
        self.bits_written = 32;
    }

    /// go back and fill in the number of bits in the page
    #[inline(always)]
    fn write_header(&mut self, header: u32) {
        let bits_written_header = header.to_le_bytes();
        // Safe version:
        // self.bytes[..bits_written_header.len()].copy_from_slice(&bits_written_header);
        unsafe {
            copy_nonoverlapping(
                bits_written_header.as_ptr(),
                self.bytes.as_mut_ptr(),
                bits_written_header.len(),
            );
        }
    }

    /// append a u32 value to the page
    /// this will copy as a block of bytes ignoring pending bits
    #[inline(always)]
    fn write_u32le(&mut self, value: u32) {
        debug_assert!(self.bits.len() == 0);
        let bytes = value.to_le_bytes();
        let offset = self.bits_written / 8;
        // Safe version:
        // self.bytes[offset..offset + bytes.len()].copy_from_slice(&bytes);
        unsafe {
            copy_nonoverlapping(
                bytes.as_ptr(),
                self.bytes.as_mut_ptr().add(offset),
                bytes.len(),
            );
        }
        self.bits_written += bytes.len() * 8;
    }

    /// write the symbols for an entry if there is enough room in the current page
    #[inline(always)]
    fn write_code(&mut self, code: &CodeEntry) -> bool {
        // Do not overwrite the page
        let position = self.position();
        let pending = self.bits.len();
        debug_assert!(pending < 8); // expecting less than a byte pending otherwise it should have been flushed
        if position + pending + code.bits.len() > self.page_size {
            // Flush any pending bits as this page is full
            if pending > 0 {
                // Pad the bits with zeros
                let padding = 8 - pending;
                self.bits.extend((0..padding).map(|_| 0));

                // Convert the bits to bytes
                for byte_bits in self.bits.chunks_exact(8) {
                    let byte_bits: &[usize; 8] = unsafe { byte_bits.try_into().unwrap_unchecked() };
                    let byte = unsafe {
                        PRESHIFTED7.get_unchecked(byte_bits[0])
                            | PRESHIFTED6.get_unchecked(byte_bits[1])
                            | PRESHIFTED5.get_unchecked(byte_bits[2])
                            | PRESHIFTED4.get_unchecked(byte_bits[3])
                            | PRESHIFTED3.get_unchecked(byte_bits[4])
                            | PRESHIFTED2.get_unchecked(byte_bits[5])
                            | PRESHIFTED1.get_unchecked(byte_bits[6])
                            | PRESHIFTED0.get_unchecked(byte_bits[7])
                    };
                    let offset = self.bits_written / 8;
                    #[cfg(test)]
                    {
                        self.bytes[offset] = byte;
                    }
                    #[cfg(not(test))]
                    {
                        unsafe {
                            *self.bytes.get_unchecked_mut(offset) = byte;
                        }
                    }
                    self.bits_written += 8;
                }

                // Bookkeeping
                self.bits_written -= padding;
                self.bits.clear();
            }

            return false;
        }

        // Extend bits into the page
        self.bits.extend(code.bits.iter());

        // Convert the bits to bytes
        let mut drained = 0;
        for byte_bits in self.bits.chunks_exact(8) {
            drained += 8;
            let byte_bits: &[usize; 8] = unsafe { byte_bits.try_into().unwrap_unchecked() };
            let byte = unsafe {
                PRESHIFTED7.get_unchecked(byte_bits[0])
                    | PRESHIFTED6.get_unchecked(byte_bits[1])
                    | PRESHIFTED5.get_unchecked(byte_bits[2])
                    | PRESHIFTED4.get_unchecked(byte_bits[3])
                    | PRESHIFTED3.get_unchecked(byte_bits[4])
                    | PRESHIFTED2.get_unchecked(byte_bits[5])
                    | PRESHIFTED1.get_unchecked(byte_bits[6])
                    | PRESHIFTED0.get_unchecked(byte_bits[7])
            };
            let offset = self.bits_written / 8;
            #[cfg(test)]
            {
                self.bytes[offset] = byte;
            }
            #[cfg(not(test))]
            {
                unsafe {
                    *self.bytes.get_unchecked_mut(offset) = byte;
                }
            }
            self.bits_written += 8;
        }

        // Remove emitted bits, leaving bits buffered
        self.bits.drain(..drained);

        true
    }

    /// flush the current page to nand and reset the page
    async fn flush(&mut self) -> Result<bool, E> {
        // Flush the remaining bits
        debug_assert!(self.bits.len() < 8); // expecting less than a byte pending otherwise it should have been flushed
        if !self.bits.is_empty() {
            // Pad the bits with zeros
            let padding = 8 - self.bits.len();
            self.bits.extend((0..padding).map(|_| 0));

            // Convert the bits to bytes
            for byte_bits in self.bits.chunks_exact(8) {
                let byte_bits: &[usize; 8] = unsafe { byte_bits.try_into().unwrap_unchecked() };
                let byte = unsafe {
                    PRESHIFTED7.get_unchecked(byte_bits[0])
                        | PRESHIFTED6.get_unchecked(byte_bits[1])
                        | PRESHIFTED5.get_unchecked(byte_bits[2])
                        | PRESHIFTED4.get_unchecked(byte_bits[3])
                        | PRESHIFTED3.get_unchecked(byte_bits[4])
                        | PRESHIFTED2.get_unchecked(byte_bits[5])
                        | PRESHIFTED1.get_unchecked(byte_bits[6])
                        | PRESHIFTED0.get_unchecked(byte_bits[7])
                };
                let offset = self.bits_written / 8;
                #[cfg(test)]
                {
                    self.bytes[offset] = byte;
                }
                #[cfg(not(test))]
                {
                    unsafe {
                        *self.bytes.get_unchecked_mut(offset) = byte;
                    }
                }
                self.bits_written += 8;
            }

            // Bookkeeping
            self.bits_written -= padding;
            self.bits.clear();
        }

        // The reader is going to look at the header to know how many bits to decode
        self.write_header(self.bits_written as u32);

        // Flush the bytes to NAND
        self.done |= !(*self.flush_page)(&self.bytes).await?;

        // Start the next page on a clean slate
        self.reset();

        Ok(self.done)
    }
}

/// A huffman decoder that reads successive tables and data from pages
pub struct Decoder {
    page_size: usize,
    page_threshold: usize,
    page_threshold_limit: usize,
    page_count: usize,
    state: DecodeState,
    decoder_trie: Option<CodeLookupTrie>,
    decoded_bytes: Vec<u8>,
    emitted_idx: usize,
}

/// The state while draining bytes from the decoder
#[derive(Debug, Copy, Clone)]
enum DecodeState {
    /// The decoder is reading the frequency table from NAND
    Table,
    /// The decoder is reading the encoded data from NAND
    Data,
    /// There are no more pages to read
    Done,
    /// There was an error in the encoder or malformed data
    Error,
}

impl Decoder {
    /// Create a decoder that reads pages of `page_size` bytes
    /// Every `page_threshold` pages, the decoder will rebuild the huffman tree
    pub fn new(page_size: usize, page_threshold: usize) -> Decoder {
        Decoder {
            page_size,
            page_threshold: 1,
            page_threshold_limit: page_threshold,
            page_count: 0,
            state: DecodeState::Table,
            decoder_trie: None,
            decoded_bytes: Vec::with_capacity(page_size),
            emitted_idx: 0,
        }
    }

    /// Prepare the decoder for a new round of decoding
    /// this keeps allocations, so its cheaper than a Decoder::new call
    pub fn reset(&mut self) {
        self.page_count = 0;
        self.page_threshold = 1;
        self.state = DecodeState::Table;
        self.decoder_trie = None;
        self.decoded_bytes.clear();
        self.emitted_idx = 0;
    }

    /// Drain a byte from the decoder
    pub async fn drain<E>(
        &mut self,
        input: &mut impl PageReader<E>,
    ) -> Result<Option<u8>, DecompressionError<E>> {
        // If there are already decoded bytes in the buffer, return them
        if self.emitted_idx < self.decoded_bytes.len() {
            let byte = if cfg!(test) {
                self.decoded_bytes[self.emitted_idx]
            } else {
                unsafe { *self.decoded_bytes.get_unchecked(self.emitted_idx) }
            };
            self.emitted_idx += 1;
            return Ok(Some(byte));
        }

        loop {
            match self.state {
                DecodeState::Done => {
                    return Ok(None);
                }
                DecodeState::Error => {
                    return Err(DecompressionError::Bad);
                }
                DecodeState::Table => {
                    // Read the page and check if this is the last page
                    let page = input.read_page().await?;
                    if page[..4] == [0xFF; 4] {
                        self.state = DecodeState::Done;
                        return Ok(None);
                    }

                    // Memcopy the weights from the page into the weights array
                    let mut weights = [0u32; SYMBOL_COUNT];
                    debug_assert!(page.len() >= SYMBOL_COUNT * 4 + 4); // +4 for the header
                    unsafe {
                        let weights_sz = SYMBOL_COUNT * 4; // u32s
                        let page_weights_ptr = page.as_ptr().add(4); // skip the header
                        let weights_ptr = weights.as_mut_ptr() as *mut u8;
                        core::ptr::copy_nonoverlapping(page_weights_ptr, weights_ptr, weights_sz);
                    }

                    // Build the tree from the weights
                    let root = build_tree(&weights);
                    self.decoder_trie = Some(CodeLookupTrie::new(root));

                    // Ready to start decoding data
                    self.state = DecodeState::Data;
                }
                DecodeState::Data => {
                    // Update bookkeeping
                    self.emitted_idx = 0;
                    self.decoded_bytes.clear();

                    // Read the page and check if this is the last page
                    let page = input.read_page().await?;
                    if page[..4] == [0xFF; 4] {
                        self.state = DecodeState::Done;
                        return Ok(None);
                    }

                    // Push page bits through the trie to get symbols
                    let symbol_lookup = self.decoder_trie.as_mut().unwrap();
                    let bits_written = u32::from_le_bytes(page[..4].try_into().unwrap());

                    // The number of bits written to a page must be valid
                    if !(32..=self.page_size * 8).contains(&(bits_written as usize)) {
                        self.state = DecodeState::Error;
                        return Err(DecompressionError::Bad);
                    }

                    let bytes_written = ((bits_written + 7) / 8) as usize;
                    let page_bytes = &page[4..bytes_written];

                    if !page_bytes.is_empty() {
                        let mut bits_read = 32;
                        let full_bytes = page_bytes.len() - 1;
                        for &byte in &page_bytes[..full_bytes] {
                            for i in (0..8).rev() {
                                let bit = (byte >> i) & 1;
                                if let Some(symbol) = symbol_lookup.next(bit) {
                                    self.decoded_bytes.push(symbol);
                                }
                            }
                        }
                        bits_read += full_bytes as u32 * 8;

                        // Process final byte which may be partial
                        if let Some(&last_byte) = page_bytes.last() {
                            let remaining_bits = (bits_written - bits_read) as usize;
                            for i in (0..8).rev().take(remaining_bits) {
                                let bit = (last_byte >> i) & 1;
                                if let Some(symbol) = symbol_lookup.next(bit) {
                                    self.decoded_bytes.push(symbol);
                                }
                            }
                        }
                    }

                    // If we read enough pages, the tree will be on the next page
                    self.page_count += 1;
                    if self.page_count > self.page_threshold {
                        self.page_count = 0;
                        self.page_threshold =
                            self.page_threshold_limit.min(self.page_threshold * 2);
                        self.state = DecodeState::Table;
                    } else {
                        self.state = DecodeState::Data;
                    }

                    // Emit the first byte from this page
                    // If there are already decoded bytes in the buffer, return them
                    if self.emitted_idx < self.decoded_bytes.len() {
                        let byte = if cfg!(test) {
                            self.decoded_bytes[self.emitted_idx]
                        } else {
                            unsafe { *self.decoded_bytes.get_unchecked(self.emitted_idx) }
                        };
                        self.emitted_idx += 1;
                        return Ok(Some(byte));
                    }
                }
            }
        }
    }
}

/// A reader that reads pages from NAND
#[allow(async_fn_in_trait)]
pub trait PageReader<E> {
    async fn read_page(&mut self) -> Result<&[u8], E>;
    fn reset(&mut self);
}

pub struct BufferedPageReader<E> {
    /// A page of bytes that was loaded from NAND
    bytes: Vec<u8>,
    /// A function that fills a buffer with a page from NAND
    read_page: ReadPageFutureFn<E>,
    /// Done
    done: bool,
}

/// A function that takes a mutable reference to the page and fills it with bytes from NAND
/// The future returns true if there are more pages that could be read
pub type ReadPageFutureFn<E> =
    Box<dyn for<'a> Fn(&'a mut [u8]) -> Pin<Box<dyn Future<Output = Result<bool, E>> + 'a>>>;

impl<E> BufferedPageReader<E> {
    pub fn new(page_size: usize, read_page: ReadPageFutureFn<E>) -> BufferedPageReader<E> {
        let mut bytes = Vec::with_capacity(page_size);
        unsafe { bytes.set_len(page_size) };
        BufferedPageReader {
            bytes,
            read_page,
            done: false,
        }
    }
}

impl<E> PageReader<E> for BufferedPageReader<E> {
    /// Fetch page bytes from NAND and provide a reference to them
    async fn read_page(&mut self) -> Result<&[u8], E> {
        if self.done {
            self.bytes.fill(0xFF);
            return Ok(&self.bytes);
        }
        self.done |= !(*self.read_page)(&mut self.bytes).await?;
        Ok(&self.bytes)
    }

    /// Reset the reader to start a new round of reading pages
    fn reset(&mut self) {
        self.done = false;
    }
}

/// Decompression errors can be malformed data or the error from the FutureFn
#[derive(Debug, Clone, Copy)]
pub enum DecompressionError<E> {
    /// The data is malformed or you kept calling drain after Bad occurred
    Bad,
    /// The error from the FutureFn
    Load(E),
}
impl<E> From<E> for DecompressionError<E> {
    fn from(err: E) -> Self {
        DecompressionError::Load(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::RefCell;
    use std::prelude::v1::*;
    use std::rc::Rc;
    use std::vec;
    use std::vec::Vec;

    #[test]
    fn test_std_vec() {
        let mut vec = Vec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        assert_eq!(vec, vec![1, 2, 3]);
    }

    #[test]
    fn test_flush_fn() {
        let mut buf = [1, 3, 5, 7];
        let flush: WritePageFutureFn<()> = Box::new(|page: &[u8]| {
            Box::pin(async move {
                std::dbg!("flush", page.len());
                Ok(true)
            })
        });

        smol::block_on(async {
            (*flush)(&mut buf).await.unwrap();
            assert_eq!(buf, [1, 3, 5, 7]);
        });
    }

    #[test]
    fn test_page_writer_advance() {
        let flush: WritePageFutureFn<()> = Box::new(|page| {
            Box::pin(async move {
                std::dbg!("flush", page.len());
                Ok(true)
            })
        });
        let mut wtr = BufferedPageWriter::new(2048, flush);
        smol::block_on(async {
            wtr.flush().await.unwrap();
        });
    }

    #[test]
    fn test_compress_simple() {
        let flush: WritePageFutureFn<()> = Box::new(|page| {
            Box::pin(async move {
                std::dbg!("flush", page.len());
                Ok(true)
            })
        });
        let mut wtr = BufferedPageWriter::new(2048, flush);
        let mut encoder = Encoder::new(2048, 4);

        smol::block_on(async {
            for value in 0..2048 {
                encoder.sink(value as u8, &mut wtr).await.unwrap();
            }
        });
    }

    #[test]
    fn test_compress_multi_page() {
        let flush: WritePageFutureFn<()> = Box::new(|_page| Box::pin(async move { Ok(true) }));
        let mut wtr = BufferedPageWriter::new(2048, flush);
        let mut encoder = Encoder::new(2048, 4);

        smol::block_on(async {
            for value in 0..2048 * 3 {
                encoder.sink(value as u8, &mut wtr).await.unwrap();
            }
            encoder.flush(&mut wtr).await.unwrap();
        });

        #[cfg(feature = "ratio")]
        {
            std::dbg!(
                encoder.bytes_in,
                encoder.bytes_out,
                encoder.bytes_in as f32 / encoder.bytes_out as f32
            );
        }
    }

    #[test]
    fn test_roundtrip() {
        let buf: Vec<u8> = Vec::new();
        let buf = Rc::new(RefCell::new(buf));
        let wtr_buf = buf.clone();
        let rdr_buf = buf.clone();
        let flush_page: WritePageFutureFn<()> = Box::new(move |page| {
            let buf = wtr_buf.clone();
            Box::pin(async move {
                let mut buf = buf.borrow_mut();
                buf.extend_from_slice(page);
                Ok(true)
            })
        });
        const PAGE_SIZE: usize = 2048;
        const PAGE_THRESHOLD: usize = 4;
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

        // We need to test
        // * no data
        // * less than a page
        // * exactly a page
        // * multiple pages
        // * exactly the page threshold
        // * multiple tables
        // * highly compressible data

        let bad_rand = (0..100)
            .map(|i| vec![0; i])
            .collect::<Vec<_>>()
            .into_iter()
            .map(|v| {
                let ptr = v.as_ptr();
                (ptr, v.len())
            })
            .fold(0, |acc, (ptr, len)| acc + ptr as usize * len * 31)
            % 9999991
            + 5123457;
        std::dbg!(bad_rand);

        let test_cases: Vec<Vec<u8>> = vec![
            vec![],
            (0..10).collect::<Vec<_>>(),
            (0..2048).map(|i| i as u8).collect::<Vec<_>>(),
            (0..2048 * 3).map(|i| i as u8).collect::<Vec<_>>(),
            (0..2048 * 4).map(|i| i as u8).collect::<Vec<_>>(),
            (0..1024 * 1024).map(|i| i as u8).collect::<Vec<_>>(),
            (0..bad_rand).map(|i| (31 * i) as u8).collect::<Vec<_>>(),
            (0..bad_rand)
                .map(|i| ((31 * i) % 16) as u8)
                .collect::<Vec<_>>(),
        ];

        #[cfg(feature = "ratio")]
        let mut compression_ratios = Vec::new();
        for (test_case, test_data) in test_cases.into_iter().enumerate() {
            std::dbg!(test_case);

            // Reset the encoder, writer, and decoder
            buf.borrow_mut().clear();
            encoder.reset();
            wtr.reset();
            decoder.reset();
            rdr.reset();

            // Write bytes to the encoder
            smol::block_on(async {
                for value in &test_data {
                    encoder.sink(*value, &mut wtr).await.unwrap();
                }
                encoder.flush(&mut wtr).await.unwrap();
            });

            std::dbg!(buf.borrow().len());
            let num_pages = (buf.borrow().len() + PAGE_SIZE - 1) / PAGE_SIZE;
            for page in 0..num_pages {
                let header_offset = page * PAGE_SIZE;
                let header = u32::from_le_bytes(
                    buf.borrow()[header_offset..header_offset + 4]
                        .try_into()
                        .unwrap(),
                );
                std::dbg!(header);
            }

            #[cfg(feature = "ratio")]
            {
                compression_ratios.push((
                    encoder.bytes_in as f32 / encoder.bytes_out as f32,
                    humanize_bytes::humanize_bytes_binary!(encoder.bytes_in),
                    humanize_bytes::humanize_bytes_binary!(encoder.bytes_out),
                ));
            }

            // Read bytes from the decoder
            smol::block_on(async {
                let mut idx = 0;
                while let Some(byte) = decoder.drain(&mut rdr).await.unwrap() {
                    assert_eq!(
                        byte, test_data[idx],
                        "test case {} byte {} mismatch",
                        test_case, idx
                    );
                    idx += 1;
                }
                assert_eq!(idx, test_data.len());
            });
        }

        #[cfg(feature = "ratio")]
        {
            std::dbg!(compression_ratios);
        }
    }
}
