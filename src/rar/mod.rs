pub mod bitinput;
pub mod parser;
pub mod rar3;
pub mod rar5;
pub mod rar15;
pub mod unpack15;

/// Unified view of a password-protected RAR file (RAR3, RAR5, or legacy RAR 1.5).
#[derive(Debug, Clone)]
pub struct RarInfo {
    pub version: RarVersion,
    pub rar3:    Option<rar3::Rar3Info>,
    pub rar5:    Option<rar5::Rar5Info>,
    pub rar15:   Option<rar15::Rar15Info>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RarVersion {
    Rar3,
    Rar5,
    Rar15,
}

impl RarInfo {
    pub fn encryption_name(&self) -> &'static str {
        match self.version {
            RarVersion::Rar3  => "AES-128-CBC (SHA-1 KDF, 262144 iters)",
            RarVersion::Rar5  => "AES-256-CBC (PBKDF2-HMAC-SHA256)",
            RarVersion::Rar15 => "RAR 1.5 (Crypt15 + LZH)",
        }
    }
}
