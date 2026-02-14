use crate::{Result, StereoError};

pub fn init_global_thread_pool() -> Result<()> {
    cv_core::init_global_thread_pool(None).map_err(StereoError::InvalidParameters)?;
    Ok(())
}
