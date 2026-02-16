use cv_runtime::{scheduler, GroupPolicy};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

#[test]
fn stress_test_concurrent_group_churn() {
    let s = scheduler();
    let barrier = Arc::new(Barrier::new(11)); // 10 workers + 1 main

    // Spawn 10 threads that constantly create and destroy groups
    let handles: Vec<_> = (0..10).map(|i| {
        let b = barrier.clone();
        thread::spawn(move || {
            b.wait();
            for j in 0..50 { // 50 iterations per thread
                let name = format!("churn-{}-{}", i, j);
                let policy = GroupPolicy::default(); // Shared
                
                // Create
                // Use a loop to retry if name collision happens (though names are unique here)
                if let Ok(group) = s.create_group(&name, 1, None, policy) {
                    // Submit some work
                    group.spawn(|| {
                        let _ = 2 + 2;
                    });
                    
                    // Small yield to let work start
                    thread::yield_now();
                    
                    // Destroy
                    let _ = s.remove_group(&name);
                }
            }
        })
    }).collect();

    barrier.wait(); // Start!
    
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn stress_test_heavy_load_mixing() {
    let s = scheduler();
    
    // Create an isolated group for heavy computation
    let iso_policy = GroupPolicy {
        allow_work_stealing: false,
        allow_dynamic_scaling: true,
    };
    // Use .ok() to handle case where test runs multiple times or group exists
    let _ = s.create_group("heavy-iso", 2, None, iso_policy); 
    
    // Create a shared group for lightweight tasks
    let shared_policy = GroupPolicy::default();
    let _ = s.create_group("light-shared", 4, None, shared_policy);

    let heavy_count = 50;
    let light_count = 500;
    
    let (tx, rx) = std::sync::mpsc::channel();
    let tx = Arc::new(std::sync::Mutex::new(tx));

    // Submit heavy tasks
    if let Some(g_heavy) = s.get_group("heavy-iso").unwrap() {
        for _ in 0..heavy_count {
            let tx = tx.clone();
            g_heavy.spawn(move || {
                // Simulate work (busy wait or sleep)
                std::thread::sleep(Duration::from_millis(2));
                tx.lock().unwrap().send(1).unwrap();
            });
        }
    }

    // Submit light tasks
    if let Some(g_light) = s.get_group("light-shared").unwrap() {
        for _ in 0..light_count {
            let tx = tx.clone();
            g_light.spawn(move || {
                // Trivial work
                let _ = 1 * 1;
                tx.lock().unwrap().send(1).unwrap();
            });
        }
    }

    // Verify all completed
    let mut total = 0;
    for _ in 0..(heavy_count + light_count) {
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(val) => total += val,
            Err(_) => panic!("Timed out waiting for tasks to complete"),
        }
    }
    
    assert_eq!(total, heavy_count + light_count);
}
