//! Deterministic RNG backed by a NIST SP 800-90A Rev. 1 CTR_DRBG
//! (AES-256), via the `nist-drbg-rs` crate, feature `aes-ctr`.
//!
//! This replaces the old xorshift64 stream. The old stream was F2-linear
//! with 64 bits of state, so every R vector it produced was a window of
//! one linear sequence and all R vectors together spanned at most 64
//! dimensions over GF(1024) -- fatal once w > 64. CTR_DRBG output is
//! AES-based and nonlinear, so streams from different seeds, and windows
//! of one stream, do not share that linear cap.
//!
//! No ChaCha, no DJB construction: CTR_DRBG/AES is a NIST-approved
//! mechanism (SP 800-90A), and AES is the fast path here (AES-NI where
//! available).
//!
//! Determinism: a `Rng` is fully determined by its u64 seed. The seed is
//! *not* claimed to be 256 bits of entropy -- it is a test seed. It is
//! expanded to a 32-byte entropy input (seed bytes repeated) plus a
//! 16-byte nonce (seed bytes, zero-padded), with a fixed personalization
//! string, instantiated via `new_with_df` (with the CTR_DRBG derivation
//! function). The derivation function then diffuses small seed
//! differences nonlinearly, unlike xorshift, where seed 42 and seed 43
//! gave linearly related streams.
//!
//! `below(n)` uses rejection sampling, not `% n`, so there is no modulo
//! bias for any n.

use nist_drbg_rs::{AesCtr256Drbg, Drbg, Policy};

const BUF_LEN: usize = 1024;

pub struct Rng {
    pub seed: u64,
    drbg: AesCtr256Drbg,
    buf: [u8; BUF_LEN],
    pos: usize,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Seed 0 used to be a stuck state for xorshift; for CTR_DRBG it
        // is a valid (if low-entropy) seed, but keep the old guard's
        // intent: never instantiate with an all-zero effective seed.
        let eff = if seed == 0 {
            0x2545_F491_4F6C_DD1D
        } else {
            seed
        };
        let seed_bytes = eff.to_le_bytes();
        let mut entropy = [0u8; 32];
        for chunk in entropy.chunks_mut(8) {
            chunk.copy_from_slice(&seed_bytes);
        }
        let mut nonce = [0u8; 16];
        nonce[..8].copy_from_slice(&seed_bytes);
        let drbg = AesCtr256Drbg::new_with_df(&entropy, &nonce, b"rlce-test-v1", Policy::default())
            .expect("CTR_DRBG instantiate with derivation function");
        let mut r = Rng {
            seed: eff,
            drbg,
            buf: [0u8; BUF_LEN],
            pos: BUF_LEN,
        };
        r.refill();
        r
    }

    fn refill(&mut self) {
        self.drbg
            .generate(&mut self.buf)
            .expect("CTR_DRBG generate (reseed interval not reached in test use)");
        self.pos = 0;
    }

    pub fn next(&mut self) -> u64 {
        if self.pos + 8 > BUF_LEN {
            self.refill();
        }
        let mut b = [0u8; 8];
        b.copy_from_slice(&self.buf[self.pos..self.pos + 8]);
        self.pos += 8;
        u64::from_le_bytes(b)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        if n == 1 {
            return 0;
        }
        // Rejection sampling: largest multiple of n below 2^64.
        let limit = (u64::MAX / n) * n;
        loop {
            let v = self.next();
            if v < limit {
                return v % n;
            }
        }
    }

    /// Random subset of size `sub` from `0..pop` (partial Fisher-Yates).
    pub fn subset(&mut self, pop: usize, sub: usize) -> Vec<usize> {
        assert!(sub <= pop);
        let mut idx: Vec<usize> = (0..pop).collect();
        for i in 0..sub {
            let j = i + self.below((pop - i) as u64) as usize;
            idx.swap(i, j);
        }
        idx.truncate(sub);
        idx
    }
}
