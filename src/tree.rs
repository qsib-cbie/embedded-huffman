use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::mem::transmute;
use core::pin::Pin;

use crate::SYMBOL_COUNT;

/// A list of bits that represent a symbol in the Huffman tree
#[derive(Default, Debug, Clone)]
pub struct CodeEntry {
    pub bits: Vec<u8>,
}

pub enum Node {
    Leaf(LeafNode),
    Internal(InternalNode),
}

impl Node {
    fn weight(&self) -> usize {
        match self {
            Node::Leaf(leaf) => leaf.weight,
            Node::Internal(internal) => internal.weight,
        }
    }
}

pub struct LeafNode {
    weight: usize,
    word: u8,
}

pub struct InternalNode {
    weight: usize,
    left: Box<Node>,
    right: Box<Node>,
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.weight().cmp(&other.weight()))
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight().cmp(&other.weight())
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.weight() == other.weight()
    }
}

impl Eq for Node {}

// Build the Huffman tree from the symbol frequencies
pub fn build_tree(weights: &[u32; SYMBOL_COUNT]) -> Node {
    // Create leafs in a buffer with room up to the max number of nodes
    let mut nodes = heapless::BinaryHeap::<Node, heapless::binary_heap::Min, SYMBOL_COUNT>::new();

    // Safe version would require Debug bloat and too many ifs
    unsafe {
        for (word, &weight) in weights.iter().enumerate() {
            let node = Node::Leaf(LeafNode {
                weight: weight as usize,
                word: word as u8,
            });
            nodes.push_unchecked(node);
        }

        // Pop the two smallest nodes off to make internal nodes until we only have a root
        while nodes.len() > 1 {
            let left = nodes.pop_unchecked();
            let right = nodes.pop_unchecked();
            nodes.push_unchecked(Node::Internal(InternalNode {
                weight: left.weight() + right.weight(),
                left: Box::new(left),
                right: Box::new(right),
            }));
        }

        nodes.pop_unchecked()
    }
}

// Build the code table for bit representation of the tree
pub fn build_code_table(root: Node, code_table: &mut [CodeEntry]) {
    let code_table: &mut [CodeEntry; SYMBOL_COUNT] =
        (&mut code_table[..SYMBOL_COUNT]).try_into().unwrap();

    // Build the code table
    let mut to_visit = VecDeque::with_capacity(SYMBOL_COUNT * 2 - 1);
    to_visit.push_back((root, vec![]));
    while let Some((node, mut bits)) = to_visit.pop_front() {
        match node {
            Node::Leaf(leaf) => {
                code_table[leaf.word as usize].bits = bits;
            }
            Node::Internal(internal) => {
                bits.reserve(1);
                let mut left_bits = bits.clone();
                let mut right_bits = bits;
                left_bits.push(0);
                right_bits.push(1);
                to_visit.push_back((*internal.left, left_bits));
                to_visit.push_back((*internal.right, right_bits));
            }
        }
    }

    // Drop unused space, code tables live a long time
    for entry in code_table {
        entry.bits.shrink_to_fit();
    }
}

/// A structure for iterating bit by bit through consecutive CodeEntries
pub struct CodeLookupTrie {
    /// the huffman tree,
    node: Pin<Box<Node>>,
    /// the current node
    /// the lifetime is transmuted as this will be a self reference
    node_ref: &'static Node,
}

impl CodeLookupTrie {
    /// Create a new decoding trie that traverses the Huffman tree
    pub fn new(root: Node) -> CodeLookupTrie {
        let node = Box::pin(root);
        let node_ref = unsafe { transmute(&*node) };
        CodeLookupTrie { node, node_ref }
    }

    /// Take a bit (0 or 1) and traverse the Huffman tree.
    /// Returns Some(symbol) if we reach a leaf node, or None if we're still traversing.
    ///
    /// This allows us to decode a stream of bits into symbols by calling next() repeatedly
    /// with each bit in the stream. When we reach a leaf node, we get the decoded symbol
    /// and automatically reset back to the root for the next symbol.
    #[inline(always)]
    pub fn next(&mut self, bit: u8) -> Option<u8> {
        debug_assert!(bit == 0 || bit == 1); // too expensive in release mode

        // The current node is always an internal node
        let Node::Internal(node) = self.node_ref else {
            unreachable!()
        };

        // If 0, go left, if 1 go right
        let node = if bit == 0 { &node.left } else { &node.right };

        // If leaf return the symbol, go back to root
        let node: &Node = node.as_ref();
        match node {
            Node::Leaf(leaf) => {
                let root: &Node = &*self.node.as_ref();
                self.node_ref = unsafe { transmute(root) };
                Some(leaf.word)
            }
            Node::Internal(_) => {
                self.node_ref = unsafe { transmute(node) };
                None
            }
        }
    }
}
