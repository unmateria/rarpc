pub mod bruteforce;
pub mod combinator;
pub mod engine;
pub mod markov;
pub mod mask;
pub mod rules;
pub mod source;
pub mod wordlist;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// How passwords should be generated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttackMode {
    Wordlist(PathBuf),
    BruteForce {
        charset: String,
        min_len: usize,
        max_len: usize,
    },
    Mask(String),
    /// Wordlist `wordlist` expanded through a rulefile.
    Rules {
        wordlist: PathBuf,
        rules:    PathBuf,
    },
    /// Combinator: emits w_i and w_i||w_j for every i,j. With `triple`,
    /// also emits w_i||w_j||w_k. Optional rules file layered on top.
    Combinator {
        wordlist: PathBuf,
        rules:    Option<PathBuf>,
        triple:   bool,
    },
    /// Char-level trigram model trained with `rarpc train-markov`.
    Markov {
        model:   PathBuf,
        min_len: usize,
        max_len: usize,
    },
}

/// Full attack configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackConfig {
    pub mode: AttackMode,
    pub batch_size: u32,
    pub gpu_index: usize,
    pub num_gpus: usize,
    pub cpu_only: bool,
}

/// A batch of password candidates ready for dispatch
pub struct PasswordBatch {
    /// Passwords as raw bytes (UTF-8 for RAR5, UTF-16LE for RAR3)
    pub passwords: Vec<Vec<u8>>,
    /// Offset into the global search space (for session resume)
    pub start_position: u64,
}
