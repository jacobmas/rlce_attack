//! Binary key serialization for the end-to-end attack.
//!
//! Two files:
//!   - `key.pub`: the PUBLIC key (what the attacker sees). Columns only.
//!   - `key.meta`: SECRET metadata (twin pairing, purity) for TESTING only.
//!     The attack subcommands never read this; `verify` uses it to score.
//!
//! Format (all integers little-endian):
//!   key.pub:  magic "RLCEPK01" (8B) | q u32 | n u32 | k u32 | w u32
//!             | N=n+w columns, each k u16s
//!   key.meta: magic "RLCEMT01" (8B) | q u32 | n u32 | k u32 | w u32
//!             | is_pure: N bytes (0/1)
//!             | pair_id: N i32 (-1 if pure, else 0..w-1)
//!             | pairs: w pairs of (u32, u32)
//!
//! Streaming: columns are read/written sequentially. For multi-GB keys the
//! OS page cache handles it; we never need more than the columns themselves
//! in RAM (N*k*2 bytes).

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

pub const PUB_MAGIC: &[u8; 8] = b"RLCEPK01";
pub const META_MAGIC: &[u8; 8] = b"RLCEMT01";

/// Public key as seen by the attacker.
pub struct PublicKey {
    pub q: u32,
    pub n: usize,
    pub k: usize,
    pub w: usize,
    pub n_pub: usize,
    /// N columns, each a Vec<u16> of length k.
    pub cols: Vec<Vec<u16>>,
}

/// Secret metadata for testing/scoring (NOT available to the attacker).
pub struct KeyMeta {
    pub q: u32,
    pub n: usize,
    pub k: usize,
    pub w: usize,
    pub is_pure: Vec<bool>,
    /// pair index 0..w-1, or None if pure.
    pub pair_id: Vec<Option<usize>>,
    pub pairs: Vec<(usize, usize)>,
}

fn w_u32(w: &mut impl Write, x: u32) -> std::io::Result<()> {
    w.write_all(&x.to_le_bytes())
}
fn w_u16(w: &mut impl Write, x: u16) -> std::io::Result<()> {
    w.write_all(&x.to_le_bytes())
}
fn r_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn r_u16(r: &mut impl Read) -> std::io::Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

/// Write the public key file.
pub fn write_pub(path: &Path, q: u32, n: usize, k: usize, w: usize, cols: &[Vec<u16>]) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w_ = BufWriter::new(f);
    w_.write_all(PUB_MAGIC)?;
    w_u32(&mut w_, q)?;
    w_u32(&mut w_, n as u32)?;
    w_u32(&mut w_, k as u32)?;
    w_u32(&mut w_, w as u32)?;
    // columns sequentially (streaming-friendly)
    for col in cols {
        debug_assert_eq!(col.len(), k);
        for &x in col {
            w_u16(&mut w_, x)?;
        }
    }
    w_.flush()
}

/// Read the public key file.
pub fn read_pub(path: &Path) -> std::io::Result<PublicKey> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != PUB_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad public key magic",
        ));
    }
    let q = r_u32(&mut r)?;
    let n = r_u32(&mut r)? as usize;
    let k = r_u32(&mut r)? as usize;
    let w = r_u32(&mut r)? as usize;
    let n_pub = n + w;
    let mut cols = Vec::with_capacity(n_pub);
    for _ in 0..n_pub {
        let mut col = vec![0u16; k];
        for i in 0..k {
            col[i] = r_u16(&mut r)?;
        }
        cols.push(col);
    }
    Ok(PublicKey { q, n, k, w, n_pub, cols })
}

/// Write the secret metadata file (testing only).
pub fn write_meta(
    path: &Path,
    q: u32,
    n: usize,
    k: usize,
    w: usize,
    is_pure: &[bool],
    pair_id: &[Option<usize>],
    pairs: &[(usize, usize)],
) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w_ = BufWriter::new(f);
    w_.write_all(META_MAGIC)?;
    w_u32(&mut w_, q)?;
    w_u32(&mut w_, n as u32)?;
    w_u32(&mut w_, k as u32)?;
    w_u32(&mut w_, w as u32)?;
    let n_pub = n + w;
    debug_assert_eq!(is_pure.len(), n_pub);
    for &p in is_pure {
        w_.write_all(&[if p { 1 } else { 0 }])?;
    }
    for pid in pair_id {
        let v: i32 = pid.map(|x| x as i32).unwrap_or(-1);
        w_.write_all(&v.to_le_bytes())?;
    }
    debug_assert_eq!(pairs.len(), w);
    for &(a, b) in pairs {
        w_u32(&mut w_, a as u32)?;
        w_u32(&mut w_, b as u32)?;
    }
    w_.flush()
}

/// Read the secret metadata file.
pub fn read_meta(path: &Path) -> std::io::Result<KeyMeta> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != META_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad meta magic",
        ));
    }
    let q = r_u32(&mut r)?;
    let n = r_u32(&mut r)? as usize;
    let k = r_u32(&mut r)? as usize;
    let w = r_u32(&mut r)? as usize;
    let n_pub = n + w;
    let mut is_pure = vec![false; n_pub];
    for i in 0..n_pub {
        let mut b = [0u8; 1];
        r.read_exact(&mut b)?;
        is_pure[i] = b[0] != 0;
    }
    let mut pair_id = vec![None; n_pub];
    for i in 0..n_pub {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        let v = i32::from_le_bytes(b);
        if v >= 0 {
            pair_id[i] = Some(v as usize);
        }
    }
    let mut pairs = Vec::with_capacity(w);
    for _ in 0..w {
        let a = r_u32(&mut r)? as usize;
        let b = r_u32(&mut r)? as usize;
        pairs.push((a, b));
    }
    Ok(KeyMeta { q, n, k, w, is_pure, pair_id, pairs })
}
