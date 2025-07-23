use log::{debug, error, info, warn};

fn main() {
    env_logger::Builder::new() 
        .filter_level(log::LevelFilter::Debug)
        .init();
    
    debug!("This is a debug log");
    info!("This is an info log");
    warn!("This is a warning log");
    error!("This is an error log");
    
}