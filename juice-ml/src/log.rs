#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => ({
        println!("⚠️ {}", format!($($arg)*));
    })
}

#[macro_export]
macro_rules! retry {
    ($($arg:tt)*) => ({
        println!("🔧 {}", format!($($arg)*));
    })
}

#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => ({
        println!("ℹ️ {}", format!($($arg)*));
    })
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => ({
        println!("⛔ {}", format!($($arg)*));
    })
}

#[macro_export]
macro_rules! sparkle {
    ($($arg:tt)*) => ({
        println!("✨ {}", format!($($arg)*));
    })
}
