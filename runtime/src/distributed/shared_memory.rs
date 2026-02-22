use super::LoadCoordinator;
use cv_hal::DeviceId;
use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

const MAX_PROCESSES: usize = 64;

#[repr(C)]
struct LoadSlot {
    lock: AtomicU32,
    device_id: AtomicU32,
    load: AtomicU32,
    timestamp: AtomicU64,
    pid: AtomicU32,
}

impl Default for LoadSlot {
    fn default() -> Self {
        Self {
            lock: AtomicU32::new(0),
            device_id: AtomicU32::new(0),
            load: AtomicU32::new(0),
            timestamp: AtomicU64::new(0),
            pid: AtomicU32::new(0),
        }
    }
}

impl LoadSlot {
    fn try_lock(&self) -> bool {
        self.lock
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    fn unlock(&self) {
        self.lock.store(0, Ordering::Release);
    }

    fn with_lock<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Self) -> R,
    {
        let mut spins = 0;
        while !self.try_lock() {
            spins += 1;
            if spins > 1000 {
                std::hint::spin_loop();
            }
            std::thread::yield_now();
        }
        let result = f(self);
        self.unlock();
        result
    }
}

#[repr(C)]
struct ShmHeader {
    magic: AtomicU32,
    version: AtomicU32,
    num_slots: AtomicU32,
    slot_size: AtomicU32,
}

pub struct ShmCoordinator {
    mmap: memmap2::MmapMut,
    slot_index: usize,
    #[allow(dead_code)]
    pid: u32,
}

impl ShmCoordinator {
    pub fn new(name: &str, size: usize) -> io::Result<Self> {
        let path = Self::get_shm_path(name);

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;

        file.set_len(size as u64)?;

        let mut mmap = unsafe { memmap2::MmapOptions::new().map_mut(&file)? };

        let header = Self::header_mut(&mut mmap);
        let magic = 0x43565254;

        if header.magic.load(Ordering::Relaxed) != magic {
            header.magic.store(magic, Ordering::Relaxed);
            header.version.store(1, Ordering::Relaxed);
            header
                .num_slots
                .store(MAX_PROCESSES as u32, Ordering::Relaxed);
            header
                .slot_size
                .store(std::mem::size_of::<LoadSlot>() as u32, Ordering::Relaxed);
        }

        let pid = std::process::id();
        let slot_index = Self::find_or_allocate_slot(&mut mmap, pid)?;

        Ok(Self {
            mmap,
            slot_index,
            pid,
        })
    }

    fn get_shm_path(name: &str) -> std::path::PathBuf {
        #[cfg(target_os = "linux")]
        {
            std::path::PathBuf::from(format!("/dev/shm/{}", name))
        }
        #[cfg(not(target_os = "linux"))]
        {
            std::env::temp_dir().join(format!("cv_runtime_{}", name))
        }
    }

    fn header_mut(mmap: &mut memmap2::MmapMut) -> &mut ShmHeader {
        unsafe { &mut *(mmap.as_mut_ptr() as *mut ShmHeader) }
    }

    fn find_or_allocate_slot(mmap: &mut memmap2::MmapMut, pid: u32) -> io::Result<usize> {
        let header = Self::header_mut(mmap);
        let num_slots = header.num_slots.load(Ordering::Relaxed) as usize;
        let header_size = std::mem::size_of::<ShmHeader>();
        let slot_size = std::mem::size_of::<LoadSlot>();

        for i in 0..num_slots.min(MAX_PROCESSES) {
            let offset = header_size + i * slot_size;
            let slot = unsafe { &mut *(mmap.as_mut_ptr().add(offset) as *mut LoadSlot) };

            let slot_pid = slot.pid.load(Ordering::Relaxed);

            if slot_pid == pid {
                return Ok(i);
            }

            if slot_pid == 0 || !Self::is_process_alive(slot_pid) {
                if slot.try_lock() {
                    let current_pid = slot.pid.load(Ordering::Relaxed);
                    if current_pid == 0 || !Self::is_process_alive(current_pid) {
                        slot.pid.store(pid, Ordering::Relaxed);
                        slot.unlock();
                        return Ok(i);
                    }
                    slot.unlock();
                }
            }
        }

        Err(io::Error::new(
            io::ErrorKind::Other,
            "No available slots in shared memory coordinator",
        ))
    }

    fn get_slot(&self) -> &LoadSlot {
        let header_size = std::mem::size_of::<ShmHeader>();
        let slot_size = std::mem::size_of::<LoadSlot>();
        let offset = header_size + self.slot_index * slot_size;

        unsafe { &*(self.mmap.as_ptr().add(offset) as *const LoadSlot) }
    }

    fn is_process_alive(pid: u32) -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new(&format!("/proc/{}", pid)).exists()
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = pid;
            true
        }
    }

    fn current_timestamp() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

impl LoadCoordinator for ShmCoordinator {
    fn update_load(&self, device_load: &HashMap<DeviceId, usize>) -> std::io::Result<()> {
        let timestamp = Self::current_timestamp();

        for (&device_id, &load) in device_load {
            let slot = self.get_slot();
            slot.with_lock(|s| {
                s.device_id.store(device_id.0, Ordering::Relaxed);
                s.load.store(load as u32, Ordering::Relaxed);
                s.timestamp.store(timestamp, Ordering::Relaxed);
            });
        }

        Ok(())
    }

    fn get_global_load(&self) -> std::io::Result<HashMap<DeviceId, usize>> {
        let mut aggregate = HashMap::new();
        let now = Self::current_timestamp();
        let stale_timeout = 30_000u64;

        let header = unsafe { &*(self.mmap.as_ptr() as *const ShmHeader) };
        let num_slots = header.num_slots.load(Ordering::Relaxed) as usize;
        let header_size = std::mem::size_of::<ShmHeader>();
        let slot_size = std::mem::size_of::<LoadSlot>();

        for i in 0..num_slots {
            let offset = header_size + i * slot_size;
            let slot = unsafe { &*(self.mmap.as_ptr().add(offset) as *const LoadSlot) };

            let slot_pid = slot.pid.load(Ordering::Relaxed);
            if slot_pid == 0 {
                continue;
            }

            let timestamp = slot.timestamp.load(Ordering::Relaxed);
            if now.saturating_sub(timestamp) > stale_timeout {
                if !Self::is_process_alive(slot_pid) {
                    continue;
                }
            }

            let device_id = slot.device_id.load(Ordering::Relaxed);
            let load = slot.load.load(Ordering::Relaxed);

            if device_id != 0 {
                *aggregate.entry(DeviceId(device_id)).or_insert(0) += load as usize;
            }
        }

        Ok(aggregate)
    }

    fn cleanup(&self) {
        let slot = self.get_slot();
        slot.with_lock(|s| {
            s.pid.store(0, Ordering::Relaxed);
            s.device_id.store(0, Ordering::Relaxed);
            s.load.store(0, Ordering::Relaxed);
            s.timestamp.store(0, Ordering::Relaxed);
        });
    }
}

impl Drop for ShmCoordinator {
    fn drop(&mut self) {
        self.cleanup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shm_coordinator() {
        let name = format!("test_shm_{}", std::process::id());
        let coord = ShmCoordinator::new(&name, 4096).unwrap();

        let mut load = HashMap::new();
        load.insert(DeviceId(1), 10);

        coord.update_load(&load).unwrap();

        let global = coord.get_global_load().unwrap();
        assert!(global.contains_key(&DeviceId(1)));
    }
}
