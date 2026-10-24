use criterion::{Criterion, black_box, criterion_group, criterion_main};
use esag::distribution::Esag;
use esag::loglik::{loglik_eta, loglik_eta_grad};
use esag::regression::EsagRegressor;
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn bench(c: &mut Criterion) {
    for d in [3usize, 4, 5] {
        let b = esag::dims::eta_len(d);
        let eta: Vec<f64> = (0..b).map(|i| 0.3 + (i as f64 * 0.77).sin()).collect();
        let raw: Vec<f64> = (0..d).map(|i| (i as f64 * 1.9).cos()).collect();
        let n = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
        let y: Vec<f64> = raw.iter().map(|v| v / n).collect();
        let mut g = vec![0.0; b];
        c.bench_function(&format!("loglik d={d}"), |bn| {
            bn.iter(|| loglik_eta::<f64>(black_box(&eta), black_box(&y)))
        });
        c.bench_function(&format!("loglik+grad d={d}"), |bn| {
            bn.iter(|| loglik_eta_grad(black_box(&eta), black_box(&y), &mut g))
        });
    }
    // full regression fit, d = 3, n = 1000, one covariate
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let n = 1000;
    let x = DMatrix::from_fn(n, 1, |i, _| ((i as f64) * 0.37).sin());
    let mut y = DMatrix::zeros(n, 3);
    let mut buf = [0.0; 3];
    for i in 0..n {
        let xi = x[(i, 0)];
        let dist = Esag::new(
            &[1.5 - 0.6 * xi, 0.5 + 1.2 * xi, 1.0 + 0.3 * xi],
            &[0.8 + 0.5 * xi, -0.4 + 0.7 * xi],
        )
        .unwrap();
        let fr = dist.frame();
        dist.sample_one(&fr, &mut rng, &mut buf);
        for k in 0..3 {
            y[(i, k)] = buf[k];
        }
    }
    c.bench_function("fit d=3 n=1000 p=1 (no hessian)", |bn| {
        bn.iter(|| {
            EsagRegressor::new()
                .with_hessian(false)
                .fit(black_box(&x), black_box(&y))
                .unwrap()
        })
    });
    c.bench_function("fit d=3 n=1000 p=1 (with hessian)", |bn| {
        bn.iter(|| {
            EsagRegressor::new()
                .fit(black_box(&x), black_box(&y))
                .unwrap()
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
