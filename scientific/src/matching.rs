//! Optimal Bipartite Matching (Hungarian Algorithm / Munkres)
//!
//! Solves the assignment problem: given a cost matrix, find the minimum cost assignment
//! of rows to columns such that each row is assigned to at most one column.

/// Solves the assignment problem for a square or rectangular cost matrix.
/// Returns a list of (row, col) assignments.
pub fn hungarian_matching(cost_matrix: &Vec<Vec<f64>>) -> Vec<(usize, usize)> {
    if cost_matrix.is_empty() || cost_matrix[0].is_empty() {
        return Vec::new();
    }

    let rows = cost_matrix.len();
    let cols = cost_matrix[0].len();
    let n = rows.max(cols);

    // 1. Pad matrix to square if needed
    let mut matrix = vec![vec![0.0; n]; n];
    let mut max_val = 0.0;
    for r in 0..rows {
        for c in 0..cols {
            matrix[r][c] = cost_matrix[r][c];
            if matrix[r][c] > max_val { max_val = matrix[r][c]; }
        }
    }
    // For rectangular padding, we fill with a large value to avoid fake matches
    // unless we want to match everything (then 0 is fine).
    // In CV tracking, we usually pad with large value.
    for r in 0..n {
        for c in 0..n {
            if r >= rows || c >= cols {
                matrix[r][c] = max_val * 10.0 + 100.0;
            }
        }
    }

    // --- Full Munkres Algorithm (Step-by-Step) ---
    
    // Step 1: Row reduction
    for r in 0..n {
        let min = matrix[r].iter().cloned().fold(f64::INFINITY, f64::min);
        for c in 0..n { matrix[r][c] -= min; }
    }

    // Step 2: Initial starring
    let mut mask = vec![vec![0u8; n]; n]; // 0: normal, 1: starred, 2: primed
    let mut row_covered = vec![false; n];
    let mut col_covered = vec![false; n];

    for r in 0..n {
        for c in 0..n {
            if !row_covered[r] && !col_covered[c] && matrix[r][c].abs() < 1e-9 {
                mask[r][c] = 1;
                row_covered[r] = true;
                col_covered[c] = true;
            }
        }
    }
    // Reset covers
    row_covered.fill(false);
    col_covered.fill(false);

    let mut step = 3;
    loop {
        match step {
            3 => {
                // Step 3: Cover columns containing a starred zero
                let mut count = 0;
                for r in 0..n {
                    for c in 0..n {
                        if mask[r][c] == 1 { col_covered[c] = true; }
                    }
                }
                for c in 0..n { if col_covered[c] { count += 1; } }
                
                if count >= n { break; } // Done!
                step = 4;
            }
            4 => {
                // Step 4: Find a non-covered zero and prime it
                if let Some((r, c)) = find_uncovered_zero(&matrix, &row_covered, &col_covered, n) {
                    mask[r][c] = 2;
                    // Find starred zero in this row
                    if let Some(star_c) = find_star_in_row(&mask, r, n) {
                        row_covered[r] = true;
                        col_covered[star_c] = false;
                        step = 4;
                    } else {
                        // Step 5: Augmenting path
                        mask = step5(&mask, r, c, n);
                        row_covered.fill(false);
                        col_covered.fill(false);
                        mask_clear_primes(&mut mask, n);
                        step = 3;
                    }
                } else {
                    // Step 6: Modify weights
                    step6(&mut matrix, &row_covered, &col_covered, n);
                    step = 4;
                }
            }
            _ => break,
        }
    }

    let mut assignments = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            if mask[r][c] == 1 {
                assignments.push((r, c));
            }
        }
    }
    assignments
}

fn find_uncovered_zero(m: &Vec<Vec<f64>>, rc: &[bool], cc: &[bool], n: usize) -> Option<(usize, usize)> {
    for r in 0..n {
        for c in 0..n {
            if !rc[r] && !cc[c] && m[r][c].abs() < 1e-9 { return Some((r, c)); }
        }
    }
    None
}

fn find_star_in_row(mask: &Vec<Vec<u8>>, r: usize, n: usize) -> Option<usize> {
    for c in 0..n { if mask[r][c] == 1 { return Some(c); } }
    None
}

fn find_star_in_col(mask: &Vec<Vec<u8>>, c: usize, n: usize) -> Option<usize> {
    for r in 0..n { if mask[r][c] == 1 { return Some(r); } }
    None
}

fn find_prime_in_row(mask: &Vec<Vec<u8>>, r: usize, n: usize) -> Option<usize> {
    for c in 0..n { if mask[r][c] == 2 { return Some(c); } }
    None
}

fn mask_clear_primes(mask: &mut Vec<Vec<u8>>, n: usize) {
    for r in 0..n { for c in 0..n { if mask[r][c] == 2 { mask[r][c] = 0; } } }
}

fn step5(mask: &Vec<Vec<u8>>, r0: usize, c0: usize, n: usize) -> Vec<Vec<u8>> {
    let mut new_mask = mask.clone();
    let mut path = vec![(r0, c0)];
    
    let mut curr_r = r0;
    let mut curr_c = c0;
    
    loop {
        if let Some(r) = find_star_in_col(&new_mask, curr_c, n) {
            path.push((r, curr_c));
            curr_r = r;
        } else { break; }
        
        if let Some(c) = find_prime_in_row(&new_mask, curr_r, n) {
            path.push((curr_r, c));
            curr_c = c;
        } else { break; }
    }
    
    for (r, c) in path {
        if new_mask[r][c] == 1 { new_mask[r][c] = 0; }
        else { new_mask[r][c] = 1; }
    }
    new_mask
}

fn step6(m: &mut Vec<Vec<f64>>, rc: &[bool], cc: &[bool], n: usize) {
    let mut min = f64::INFINITY;
    for r in 0..n {
        for c in 0..n {
            if !rc[r] && !cc[c] { min = min.min(m[r][c]); }
        }
    }
    
    for r in 0..n {
        for c in 0..n {
            if rc[r] { m[r][c] += min; }
            if !cc[c] { m[r][c] -= min; }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hungarian_simple() {
        let cost = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        let matches = hungarian_matching(&cost);
        assert_eq!(matches.len(), 3);
        
        let mut total_cost = 0.0;
        for (r, c) in matches {
            total_cost += cost[r][c];
        }
        // Min cost should be 5.0
        assert_eq!(total_cost, 5.0);
    }
}
