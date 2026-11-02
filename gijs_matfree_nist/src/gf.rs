//! GF(2^m) with log/exp tables.
//!
//! Primitive polynomials (low part, i.e. without the x^m term):
//! - m=5  (q=32):   x^5 + x^2 + 1                 -> 0b101
//! - m=6  (q=64):   x^6 + x + 1                   -> 0b011
//! - m=7  (q=128):  x^7 + x^3 + 1                 -> 0b1001
//! - m=8  (q=256):  x^8 + x^4 + x^3 + x^2 + 1     -> 0b11101
//! - m=9  (q=512):  x^9 + x^4 + 1                 -> 0b10001
//! - m=10 (q=1024): x^10 + x^3 + 1                -> 0b1001
//! - m=11 (q=2048): x^11 + x^2 + 1                -> 0b101
//! All are primitive (verified: x has order 2^m - 1 in each).
//! Any consistent representation works for the distinguisher tests;
//! this choice is documented here.

pub struct Gf {
    pub m: u32,
    pub order: usize, // 2^m
    mask: u16,
    prim_low: u16,
    exp: Vec<u16>, // size 2*(order-1), doubled to avoid mod in mul
    log: Vec<u32>, // size order, log[0] = u32::MAX
}

impl Gf {
    pub fn new(m: u32) -> Self {
        let prim_low: u16 = match m {
            5 => 0b101,     // x^5 + x^2 + 1
            6 => 0b011,     // x^6 + x + 1
            7 => 0b1001,    // x^7 + x^3 + 1
            8 => 0b11101,   // x^8 + x^4 + x^3 + x^2 + 1
            9 => 0b10001,   // x^9 + x^4 + 1
            10 => 0b1001,   // x^10 + x^3 + 1
            11 => 0b101,    // x^11 + x^2 + 1
            _ => panic!("unsupported GF(2^{}); supported m = 5..=11", m),
        };
        let order = 1usize << m;
        let mask = (order - 1) as u16;
        let mut exp = vec![0u16; 2 * (order - 1)];
        let mut log = vec![u32::MAX; order];
        let mut x: u16 = 1;
        for i in 0..(order - 1) {
            exp[i] = x;
            log[x as usize] = i as u32;
            // multiply by the primitive element (x)
            let shifted = (x as u32) << 1;
            x = if shifted & (1 << m) != 0 {
                ((shifted ^ (prim_low as u32)) & (mask as u32)) as u16
            } else {
                shifted as u16
            };
        }
        for i in 0..(order - 1) {
            exp[i + (order - 1)] = exp[i];
        }
        Self { m, order, mask, prim_low, exp, log }
    }

    pub fn for_q(q: u32) -> Self {
        assert!(q.is_power_of_two(), "q must be a power of two");
        let m = q.trailing_zeros();
        Self::new(m)
    }

    #[inline(always)]
    pub fn add(&self, a: u16, b: u16) -> u16 {
        a ^ b
    }

    #[inline(always)]
    pub fn mul(&self, a: u16, b: u16) -> u16 {
        if a == 0 || b == 0 {
            0
        } else {
            let la = self.log[a as usize] as usize;
            let lb = self.log[b as usize] as usize;
            // la+lb < 2*(order-1); exp table is doubled so no mod needed
            self.exp[la + lb]
        }
    }

    #[inline(always)]
    pub fn inv(&self, a: u16) -> u16 {
        debug_assert!(a != 0);
        let la = self.log[a as usize] as usize;
        self.exp[(self.order - 1) - la]
    }

    pub fn pow(&self, a: u16, e: usize) -> u16 {
        if e == 0 {
            return 1;
        }
        if a == 0 {
            return 0;
        }
        let la = self.log[a as usize] as usize;
        let le = (la * e) % (self.order - 1);
        self.exp[le]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_axioms_1024() {
        let gf = Gf::new(10);
        // spot-check: a * inv(a) == 1
        for a in [1u16, 2, 3, 100, 511, 1023] {
            assert_eq!(gf.mul(a, gf.inv(a)), 1);
        }
        // primitivity: alpha^(order-1) == 1, and no smaller exponent
        let alpha = 2u16; // x
        assert_eq!(gf.pow(alpha, 1023), 1);
        assert_ne!(gf.pow(alpha, 511), 1);
        // count nonzero: exp covers all
        let mut seen = vec![false; 1024];
        for i in 0..1023 {
            seen[gf.exp[i] as usize] = true;
        }
        assert!(seen[1..].iter().all(|&b| b));
    }

    #[test]
    fn field_axioms_32() {
        let gf = Gf::new(5);
        for a in [1u16, 2, 7, 31] {
            assert_eq!(gf.mul(a, gf.inv(a)), 1);
        }
        assert_eq!(gf.pow(2, 31), 1);
    }

    /// Prime factors of 2^m - 1 for each supported m.
    fn prime_factors_2m_minus_1(m: u32) -> &'static [u64] {
        match m {
            5 => &[31],          // 31 prime
            6 => &[3, 7],         // 63 = 3^2 * 7
            7 => &[127],         // 127 prime
            8 => &[3, 5, 17],    // 255 = 3 * 5 * 17
            9 => &[7, 73],       // 511 = 7 * 73
            10 => &[3, 11, 31],  // 1023 = 3 * 11 * 31
            11 => &[23, 89],     // 2047 = 23 * 89
            _ => panic!("no factorization for m={}", m),
        }
    }

    /// x (the element 2) has exact order 2^m - 1: pow(x, N) == 1 and
    /// pow(x, N/p) != 1 for every prime factor p of N.
    fn assert_primitive(m: u32) {
        let gf = Gf::new(m);
        let n: u64 = (1u64 << m) - 1;
        assert_eq!(gf.pow(2, n as usize), 1, "x^N != 1 for m={}", m);
        for &p in prime_factors_2m_minus_1(m) {
            assert_ne!(
                gf.pow(2, (n / p) as usize),
                1,
                "x^(N/{}) == 1 for m={}: not primitive",
                p,
                m
            );
        }
        // every nonzero element reachable: exp table covers F_q^*
        let mut seen = vec![false; gf.order];
        for i in 0..(gf.order - 1) {
            seen[gf.exp[i] as usize] = true;
        }
        assert!(seen[1..].iter().all(|&b| b), "exp table incomplete for m={}", m);
        // spot-check inverse
        for a in [1u16, 2, 3, (gf.order - 1) as u16] {
            assert_eq!(gf.mul(a, gf.inv(a)), 1, "inv broken for m={}", m);
        }
    }

    #[test]
    fn primitivity_all_fields() {
        for m in 5..=11 {
            assert_primitive(m);
        }
    }
}
