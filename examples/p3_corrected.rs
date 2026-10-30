use esag::distribution::Esag;
use esag::logtangent::{LtFrame, covariance};
use esag::penalized::{Grouping, PenalizedEsag};
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use std::time::Instant;

fn main() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    // ---- i.i.d., d = 6: eigenvalues of S tied in blocks {1}, {2,3,4,5}
    let d = 6;
    let mu = [2.0, -5.0, 3.0, -4.0, 5.0, 6.0];
    let fr = LtFrame::new(&mu);
    let p = d - 1;
    // S = diag(-2, 0.5, 0.5, 0.5, 0.5) rotated
    let q = nalgebra::linalg::QR::new(DMatrix::from_fn(p, p, |i, j| {
        ((i * 3 + j * 7) as f64).sin()
    }))
    .q();
    let s_mat = &q
        * DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![-2.0, 0.5, 0.5, 0.5, 0.5]))
        * q.transpose();
    let flat: Vec<f64> = (0..p)
        .flat_map(|r| (0..p).map(move |c| (r, c)))
        .map(|(r, c)| s_mat[(r, c)])
        .collect();
    let s = esag::logtangent::matrix_to_s(&flat, p);
    let v = covariance(&fr, &mu, &s);
    let dist = Esag::from_covariance(&mu, &v).unwrap();
    for n in [100usize, 400] {
        let y = dist.sample(n, &mut rng);
        let x = DMatrix::<f64>::zeros(n, 0);
        let t = Instant::now();
        let path = PenalizedEsag::eigen_gap().fit(&x, &y).unwrap();
        let b = path.best();
        println!(
            "EigenGap iid n={n}: {:.0} ms, selected λ={:.3}, active gaps {:?} (truth: [true,false,false,false]), df={}, conv={}",
            t.elapsed().as_secs_f64() * 1e3,
            b.lambda,
            b.active,
            b.df,
            b.converged
        );
        let t = Instant::now();
        let path = PenalizedEsag::isotropy().fit(&x, &y).unwrap();
        println!(
            "Isotropy iid n={n}: {:.0} ms, selected active {:?}",
            t.elapsed().as_secs_f64() * 1e3,
            path.best().active
        );
    }
    // isotropic data
    let iso = Esag::new(&mu, &vec![0.0; esag::dims::gamma_len(d)]).unwrap();
    let y = iso.sample(300, &mut rng);
    let path = PenalizedEsag::isotropy()
        .fit(&DMatrix::<f64>::zeros(300, 0), &y)
        .unwrap();
    println!(
        "Isotropy on isotropic data: active {:?} (want [false])",
        path.best().active
    );

    // ---- regression d = 8, 6 drivers: x1 mean+shape, x2 mean only, x3 shape only, x4..x6 none
    for &(d, n) in &[(7usize, 2000usize), (8, 2000)] {
        let bd = esag::dims::eta_len(d);
        let pdrv = 6;
        let lt = LtFrame::standard(d);
        let b = DMatrix::from_fn(bd, pdrv + 1, |k, j| {
            let s = (k * 7 + j * 3) as f64;
            match j {
                0 => {
                    if k < d {
                        1.0 + 0.8 * (s * 0.9).sin() + if k == d - 1 { 3.0 } else { 0.0 }
                    } else {
                        0.5 * (s * 1.3).sin()
                    }
                }
                1 => {
                    if k < d {
                        0.4 * (s * 0.7).cos()
                    } else {
                        0.25 * (s * 1.1).sin()
                    }
                }
                2 => {
                    if k < d {
                        0.4 * (s * 0.7).cos()
                    } else {
                        0.0
                    }
                }
                3 => {
                    if k < d {
                        0.0
                    } else {
                        0.25 * (s * 1.1).sin()
                    }
                }
                _ => 0.0,
            }
        });
        let x = DMatrix::from_fn(n, pdrv, |_, _| StandardNormal.sample(&mut rng));
        let mut y = DMatrix::zeros(n, d);
        let mut buf = vec![0.0; d];
        for i in 0..n {
            let eta: Vec<f64> = (0..bd)
                .map(|k| b[(k, 0)] + (0..pdrv).map(|j| b[(k, j + 1)] * x[(i, j)]).sum::<f64>())
                .collect();
            let v = covariance(&lt, &eta[..d], &eta[d..]);
            let dist = Esag::from_covariance(&eta[..d], &v).unwrap();
            dist.sample_one(&dist.frame(), &mut rng, &mut buf);
            for k in 0..d {
                y[(i, k)] = buf[k];
            }
        }
        let t = Instant::now();
        let path = PenalizedEsag::adaptive_group_lasso()
            .grouping(Grouping::CovariatePart)
            .fit(&x, &y)
            .unwrap();
        let best = path.best();
        let names: Vec<String> = (1..=pdrv)
            .flat_map(|j| [format!("x{j}.mu"), format!("x{j}.shape")])
            .collect();
        let act: Vec<String> = names
            .iter()
            .zip(&best.active)
            .filter(|(_, a)| **a)
            .map(|(n, _)| n.clone())
            .collect();
        println!(
            "Drivers d={d} n={n} ({} params): {:.2} s for {} λ, selected λ={:.2}: active {:?}\n   (truth: x1.mu x1.shape x2.mu x3.shape); relaxed refit SEs: {}",
            bd * (pdrv + 1),
            t.elapsed().as_secs_f64(),
            path.points.len(),
            best.lambda,
            act,
            path.relaxed
                .as_ref()
                .map(|r| r.std_errors().is_some())
                .unwrap_or(false)
        );
    }
}
