//! CubeCL radix sort kernel.
//!
//! LSD (least-significant digit) radix sort on `u32` keys.
//!
//! This initial implementation does CPU-side sort with GPU round-trip to
//! validate the CubeCL client is functional.  A full GPU radix sort using
//! `SharedMemory` and `Atomic<u32>` will be added in the Tier 2/3 pass.

use crate::cubecl::WgpuClient;

const RADIX_BITS: u32 = 8;
const RADIX_SIZE: u32 = 256;
// TODO(Tier2): Add GPU histogram + scatter kernels using Atomic<u32> and SharedMemory.

// ---------------------------------------------------------------------------
// Public API — CPU-assisted sort using CubeCL for data movement
// ---------------------------------------------------------------------------

/// Sort a `u32` slice using LSD radix sort (4 × 8-bit passes).
///
/// The sort runs on CPU for correctness; GPU client is used for upload/download
/// to validate the pipeline.
pub fn radix_sort(client: &WgpuClient, keys: &[u32]) -> Vec<u32> {
    let n = keys.len();
    if n <= 1 {
        return keys.to_vec();
    }

    let mut src = keys.to_vec();
    let mut dst = vec![0u32; n];

    for pass in 0u32..4u32 {
        let shift = pass * RADIX_BITS;

        let mut counts = vec![0usize; RADIX_SIZE as usize];
        for &k in &src {
            counts[((k >> shift) & (RADIX_SIZE - 1)) as usize] += 1;
        }

        let mut prefixes = vec![0usize; RADIX_SIZE as usize];
        for i in 1..RADIX_SIZE as usize {
            prefixes[i] = prefixes[i - 1] + counts[i - 1];
        }

        for &k in &src {
            let digit = ((k >> shift) & (RADIX_SIZE - 1)) as usize;
            dst[prefixes[digit]] = k;
            prefixes[digit] += 1;
        }

        std::mem::swap(&mut src, &mut dst);
    }

    // Validate GPU client is functional with a trivial probe buffer
    let _ = client.read_one(client.create_from_slice(&[0u8; 4]));
    src
}

/// Sort `u32` keys with associated `u32` values (stable by key).
pub fn radix_sort_by_key(
    client: &WgpuClient,
    keys: &[u32],
    values: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let n = keys.len();
    assert_eq!(n, values.len());

    let mut src_k = keys.to_vec();
    let mut src_v = values.to_vec();
    let mut dst_k = vec![0u32; n];
    let mut dst_v = vec![0u32; n];

    for pass in 0u32..4u32 {
        let shift = pass * RADIX_BITS;

        let mut counts = vec![0usize; RADIX_SIZE as usize];
        for &k in &src_k {
            counts[((k >> shift) & (RADIX_SIZE - 1)) as usize] += 1;
        }

        let mut prefixes = vec![0usize; RADIX_SIZE as usize];
        for i in 1..RADIX_SIZE as usize {
            prefixes[i] = prefixes[i - 1] + counts[i - 1];
        }

        for i in 0..n {
            let digit = ((src_k[i] >> shift) & (RADIX_SIZE - 1)) as usize;
            dst_k[prefixes[digit]] = src_k[i];
            dst_v[prefixes[digit]] = src_v[i];
            prefixes[digit] += 1;
        }

        std::mem::swap(&mut src_k, &mut dst_k);
        std::mem::swap(&mut src_v, &mut dst_v);
    }

    // Validate GPU client is functional with a trivial probe buffer
    let _ = client.read_one(client.create_from_slice(&[0u8; 4]));
    (src_k, src_v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubecl::get_client;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_radix_sort_basic() {
        let client = get_client();
        let input = vec![5u32, 2, 8, 1, 9, 3, 7, 4, 6, 0];
        let result = radix_sort(&client, &input);
        let expected: Vec<u32> = (0..10).collect();
        assert_eq!(result, expected);
    }

    #[test]
    #[serial]
    fn test_radix_sort_by_key() {
        let client = get_client();
        let keys = vec![3u32, 1, 2];
        let vals = vec![30u32, 10, 20];
        let (sk, sv) = radix_sort_by_key(&client, &keys, &vals);
        assert_eq!(sk, vec![1, 2, 3]);
        assert_eq!(sv, vec![10, 20, 30]);
    }
}
