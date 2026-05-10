use anyhow::{bail, Context, Result};
use std::path::Path;

use super::{RarInfo, RarVersion};
use super::rar3::{Rar3CheckMode, Rar3Info};
use super::rar5::{PswCheckData, Rar5Info};
use super::rar15::Rar15Info;

// ── Signatures ───────────────────────────────────────────────

const RAR5_SIG: &[u8] = b"Rar!\x1a\x07\x01\x00";  // 8 bytes
const RAR3_SIG: &[u8] = b"Rar!\x1a\x07\x00";       // 7 bytes

// ── Public entry point ───────────────────────────────────────

pub fn parse_rar(path: &Path) -> Result<RarInfo> {
    let data = std::fs::read(path)
        .with_context(|| format!("Cannot read file: {:?}", path))?;

    if data.starts_with(RAR5_SIG) {
        let rar5 = parse_rar5(&data)?;
        Ok(RarInfo { version: RarVersion::Rar5, rar3: None, rar5: Some(rar5), rar15: None })
    } else if data.starts_with(RAR3_SIG) {
        parse_rar3(&data)
    } else {
        bail!("Not a recognized RAR file (expected RAR3 or RAR5 signature)");
    }
}

// ── RAR5 parser ──────────────────────────────────────────────

/// RAR5 variable-length integer (little-endian, 7 bits per byte)
fn read_vint(data: &[u8], off: &mut usize) -> Result<u64> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    loop {
        if *off >= data.len() {
            bail!("vint: unexpected end of data at offset {}", *off);
        }
        let b = data[*off];
        *off += 1;
        value |= ((b & 0x7f) as u64) << shift;
        shift += 7;
        if b & 0x80 == 0 {
            break;
        }
        if shift > 63 {
            bail!("vint: value too large");
        }
    }
    Ok(value)
}

fn parse_rar5(data: &[u8]) -> Result<Rar5Info> {
    let mut off = 8usize; // skip 8-byte RAR5 signature

    loop {
        if off + 5 >= data.len() {
            bail!("RAR5: no encryption record found before end of file");
        }

        // RAR5 block layout:
        //   HeadCRC32    : 4 bytes  (uint32, not a vint)
        //   HeadSize     : vint     (size of header data starting from HeadType field)
        //   HeadType     : vint
        //   HeadFlags    : vint
        //   [optional ExtraAreaSize : vint]
        //   [optional DataAreaSize  : vint]
        //   [header-type-specific fields]

        off += 4; // skip CRC32

        // *** key fix: save off BEFORE reading header_size so we can compute header_end ***
        let head_type_start = off + {
            // peek at vint length without advancing
            let mut tmp = off;
            read_vint(data, &mut tmp)?;
            tmp - off  // bytes consumed by header_size vint
        };
        let header_size = read_vint(data, &mut off)? as usize;
        // header_end: header_size bytes are counted from HeadType field
        let header_end = off + header_size;

        if header_end > data.len() {
            bail!("RAR5: truncated header (need {} bytes, have {})", header_end, data.len());
        }

        let header_type  = read_vint(data, &mut off)?;
        let header_flags = read_vint(data, &mut off)?;

        // ExtraAreaSize: size of extra data located at END of the header.
        // Do NOT add to off now — the extra area is not at the current position.
        let extra_size: usize = if header_flags & 0x0001 != 0 {
            read_vint(data, &mut off)? as usize
        } else { 0 };

        // DataAreaSize: bytes of file data that follow the header in the stream.
        // Must be skipped when advancing to the next block.
        let data_area_size: usize = if header_flags & 0x0002 != 0 {
            read_vint(data, &mut off)? as usize
        } else { 0 };

        // Header type 4 = CRYPT (encryption header for whole archive)
        if header_type == 4 {
            // Encryption header body (RAR5 tech note §6.3):
            //   EncryptionVersion : vint (0 = AES-256)
            //   EncryptionFlags   : vint  bit 0x0001 = CRYPT_PSWCHECK
            //   Cnt               : 1 byte (log2 of PBKDF2 iteration count)
            //   Salt              : 16 bytes
            //   PswCheckData      : 12 bytes (if CRYPT_PSWCHECK set)
            //     [0..8]  = InitV  (used in SHA-256 password check)
            //     [8..12] = PswCheck (4-byte check value)

            let enc_ver   = read_vint(data, &mut off)? as u8;
            let enc_flags = read_vint(data, &mut off)?;

            if off >= data.len() {
                bail!("RAR5: truncated encryption header at cnt");
            }
            let iter_count = data[off]; off += 1;

            if off + 16 > header_end {
                bail!("RAR5: no room for 16-byte salt");
            }
            let salt: [u8; 16] = data[off..off+16].try_into().unwrap();
            off += 16;

            // PswCheckData (12 bytes) if CRYPT_PSWCHECK flag is set
            let psw_check_data = if enc_flags & 0x0001 != 0 {
                if off + 12 <= header_end {
                    let init_v: [u8; 8] = data[off..off+8].try_into().unwrap();
                    let check:  [u8; 4] = data[off+8..off+12].try_into().unwrap();
                    Some(PswCheckData { init_v, check })
                } else {
                    None
                }
            } else {
                None
            };

            return Ok(Rar5Info {
                salt,
                iv: None,  // IV for actual data decryption is per-file, not here
                iter_count,
                psw_check_data,
                enc_ver,
            });
        }

        // Header type 2 = FILE — per-file encryption lives in ExtraArea (FHEXTRA_CRYPT = 0x01)
        if header_type == 2 && extra_size > 0 && header_end >= extra_size {
            let extra_start = header_end - extra_size;
            if let Some(info) = parse_rar5_file_crypt(data, extra_start, header_end) {
                return Ok(info);
            }
        }

        // Advance past this block's header AND its data area to reach the next block.
        off = header_end + data_area_size;
    }
}

/// Scan the ExtraArea of a RAR5 FILE header for a FHEXTRA_CRYPT record (type 0x01).
/// Returns a Rar5Info if found, None otherwise.
fn parse_rar5_file_crypt(data: &[u8], extra_start: usize, extra_end: usize) -> Option<Rar5Info> {
    let mut pos = extra_start;
    while pos < extra_end {
        let mut cur = pos;

        // RecordSize vint: number of bytes for RecordType+RecordData combined
        let record_size = read_vint(data, &mut cur).ok()? as usize;
        let record_end  = cur + record_size;
        if record_end > extra_end { break; }

        // RecordType vint
        let record_type = read_vint(data, &mut cur).ok()?;

        if record_type == 0x01 {
            // FHEXTRA_CRYPT record layout (RAR5 tech note §7.3.2):
            //   EncVersion : vint (0 = AES-256)
            //   EncFlags   : vint  bit 0x0001 = CRYPT_PSWCHECK
            //   Cnt        : 1 byte  (log2 of PBKDF2 iteration count)
            //   Salt       : 16 bytes
            //   IV         : 16 bytes
            //   PswCheckData : 12 bytes (if CRYPT_PSWCHECK set)
            //     [0..8]  = InitV
            //     [8..12] = PswCheck (4-byte check value)

            let enc_ver   = read_vint(data, &mut cur).ok()? as u8;
            let enc_flags = read_vint(data, &mut cur).ok()?;

            if cur >= data.len() { return None; }
            let iter_count = data[cur]; cur += 1;

            if cur + 16 > record_end { return None; }
            let salt: [u8; 16] = data[cur..cur+16].try_into().ok()?;
            cur += 16;

            let iv: Option<[u8; 16]> = if cur + 16 <= record_end {
                let iv_bytes: [u8; 16] = data[cur..cur+16].try_into().ok()?;
                cur += 16;
                Some(iv_bytes)
            } else {
                None
            };

            let psw_check_data = if enc_flags & 0x0001 != 0 && cur + 12 <= record_end {
                let init_v: [u8; 8] = data[cur..cur+8].try_into().ok()?;
                let check:  [u8; 4] = data[cur+8..cur+12].try_into().ok()?;
                Some(PswCheckData { init_v, check })
            } else {
                None
            };

            return Some(Rar5Info { salt, iv, iter_count, psw_check_data, enc_ver });
        }

        // Skip to next record
        pos = record_end;
    }
    None
}

// ── RAR3 parser ──────────────────────────────────────────────

fn read_le16(data: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([data[off], data[off+1]])
}

fn read_le32(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]])
}

/// Try to read a valid RAR3 block header at `off`.
/// Returns (head_type, head_flags, head_size) or None if not enough data.
fn rar3_block_header(data: &[u8], off: usize) -> Option<(u8, u16, usize)> {
    if off + 7 > data.len() { return None; }
    let head_type  = data[off + 2];
    let head_flags = read_le16(data, off + 3);
    let head_size  = read_le16(data, off + 5) as usize;
    Some((head_type, head_flags, head_size))
}

/// Scan forward from `start` looking for a RAR3 file header (0x74)
/// with the ENCRYPTED flag set.  SALT flag is NOT required — old
/// RAR archives may omit it and use a global/zero salt.
/// Returns the offset of that header, or None.
fn scan_for_encrypted_file(data: &[u8], start: usize) -> Option<usize> {
    for off in start..data.len().saturating_sub(6) {
        if data[off + 2] != 0x74 { continue; }
        let flags = read_le16(data, off + 3);
        if flags & 0x0004 == 0 { continue; }   // must have ENCRYPTED flag
        let hs = read_le16(data, off + 5) as usize;
        if hs < 7 { continue; }                // minimum block size
        if off + hs > data.len() { continue; } // header must fit in file
        // Need at least 16 bytes of encrypted data after the header
        if off + hs + 16 > data.len() { continue; }
        return Some(off);
    }
    None
}

fn rar3_info(info: Rar3Info) -> RarInfo {
    RarInfo { version: RarVersion::Rar3, rar3: Some(info), rar5: None, rar15: None }
}

fn rar15_info(info: Rar15Info) -> RarInfo {
    RarInfo { version: RarVersion::Rar15, rar3: None, rar5: None, rar15: Some(info) }
}

/// Build a `Rar15Info` from a RAR 1.5 file block (HEAD_TYPE=0x74, UNP_VER=0x0f).
/// Called from both the main loop and the forward-scan fallback in `parse_rar3`.
fn parse_rar15_file_block(
    data: &[u8],
    off: usize,
    head_size: usize,
    head_flags: u16,
) -> Result<RarInfo> {
    // UNP_VER at off+24. We only accept exactly 0x0f here. RAR 2.x (0x14-0x1c)
    // uses different cipher/decompressor combos that are not yet supported.
    if off + 26 > data.len() {
        bail!("RAR 1.5: truncated file header at offset {}", off);
    }
    let unp_ver = data[off + 24];
    if unp_ver != 0x0f {
        bail!(
            "Archive uses RAR {}.{} (UNP_VER=0x{:02x}) — only RAR 1.5 (0x0f), \
             RAR 3.x (0x1d+) and RAR 5.x legacy formats are supported; \
             RAR 2.0-2.8 (Crypt20/Unpack20) is not yet implemented.",
            unp_ver / 10, unp_ver % 10, unp_ver,
        );
    }

    // Crypt15 never uses a salt. If FILE_SALT flag is set we likely
    // mis-identified the format.
    if head_flags & 0x0400 != 0 {
        bail!(
            "RAR 1.5 header claims FILE_SALT set — cipher does not use a salt, \
             file likely corrupt or mis-detected."
        );
    }

    let method    = data[off + 25];
    let pack_size = read_le32(data, off + 7);
    let unp_size  = read_le32(data, off + 11);
    let file_crc  = read_le32(data, off + 16);

    let data_start = off + head_size;
    let data_end   = data_start.checked_add(pack_size as usize)
        .ok_or_else(|| anyhow::anyhow!("RAR 1.5: pack_size overflow"))?;
    if data_end > data.len() {
        bail!(
            "RAR 1.5: packed data truncated (need {} bytes, file has {})",
            data_end, data.len(),
        );
    }
    let packed_data = data[data_start..data_end].to_vec();

    Ok(rar15_info(Rar15Info {
        packed_data,
        unp_size,
        file_crc,
        unp_ver,
        method,
    }))
}

fn parse_rar3(data: &[u8]) -> Result<RarInfo> {
    let mut off = 7usize; // skip 7-byte RAR3 signature
    // If the archive header carries a salt (ENCRYPTED_VER on 0x73), keep it
    // so file blocks without a per-file salt can use it as fallback.
    let mut archive_salt: Option<[u8; 8]> = None;

    loop {
        let (head_type, head_flags, head_size) = match rar3_block_header(data, off) {
            Some(h) => h,
            None    => break,
        };

        if head_size < 7 {
            bail!("RAR3: corrupt block at offset {} (head_size={})", off, head_size);
        }

        let next_off = off + head_size;

        match head_type {
            // ── MARK block ───────────────────────────────────────
            0x72 => {
                off = next_off.min(data.len());
            }

            // ── Archive header ───────────────────────────────────
            0x73 => {
                // ARCHIVE_HDRENCRPYT (0x0080): all subsequent headers are
                // encrypted.  The 8-byte salt lives at the fixed +13 offset
                // inside this header; the first encrypted block starts right
                // after the complete archive header (at off + head_size).
                //
                // NOTE: 0x0004 is ARCHIVE_LOCK (not encryption) — do NOT use it.
                if head_flags & 0x0080 != 0 {
                    let salt_off = off + 13;
                    if salt_off + 8 <= data.len() {
                        let salt: [u8; 8] = data[salt_off..salt_off+8].try_into().unwrap();
                        archive_salt = Some(salt);
                        // Encrypted stream begins after the full archive header.
                        let enc_off = next_off; // = off + head_size
                        let enc_block: [u8; 16] = if enc_off + 16 <= data.len() {
                            data[enc_off..enc_off+16].try_into().unwrap()
                        } else {
                            [0u8; 16]
                        };
                        return Ok(rar3_info(Rar3Info {
                            salt,
                            enc_block,
                            check_mode: Rar3CheckMode::HeadType,
                            head_type: 0x74,
                            file_crc: 0,
                            pack_size: 0,
                            auth_check: [0x74, 2],
                        }));
                    }
                }
                // Skip even if head_size is very large (embedded comment in old
                // RAR format).  Clamp so the next iteration hits end-of-data.
                off = next_off.min(data.len());
            }

            // ── File header ──────────────────────────────────────
            0x74 => {
                let encrypted = head_flags & 0x0004 != 0;
                let has_salt  = head_flags & 0x0400 != 0;

                if encrypted {
                    // UNP_VER sits at off+24 (post-fixed-fields, before
                    // METHOD at off+25). Values below 29 correspond to
                    // pre-RAR-2.9 formats that use the Crypt13 / Crypt15 /
                    // Crypt20 stream ciphers instead of AES. The present
                    // dispatch only handles RAR 3.0+; return a clear error
                    // rather than silently producing false positives via
                    // the AES heuristic.
                    if off + 25 <= data.len() {
                        let unp_ver = data[off + 24];
                        if unp_ver < 29 {
                            return parse_rar15_file_block(data, off, head_size, head_flags);
                        }
                    }

                    // Determine salt: prefer per-file salt at end of header,
                    // fall back to archive-level salt, then use zeroes.
                    let salt: [u8; 8] = if has_salt && head_size >= 8 && next_off <= data.len() {
                        data[off + head_size - 8 .. off + head_size].try_into().unwrap()
                    } else {
                        archive_salt.unwrap_or([0u8; 8])
                    };

                    // Encrypted data starts after the file header.
                    // If header extends past EOF, we still try with what we have.
                    let enc_start = next_off.min(data.len());
                    let enc_block: [u8; 16] = if enc_start + 16 <= data.len() {
                        data[enc_start..enc_start+16].try_into().unwrap()
                    } else {
                        // Not enough data — archive is truncated.
                        bail!(
                            "RAR3: encrypted file found at offset {} but data is truncated \
                             (need {} bytes, have {})",
                            off, enc_start + 16, data.len()
                        );
                    };

                    let method        = if off + 26 <= data.len() { data[off + 25] } else { 0 };
                    let file_crc      = if off + 20 <= data.len() {
                        u32::from_le_bytes(data[off+16..off+20].try_into().unwrap())
                    } else { 0 };
                    let pack_size_raw = if off + 11 <= data.len() {
                        u32::from_le_bytes(data[off+7..off+11].try_into().unwrap())
                    } else { u32::MAX };

                    let (check_mode, pack_size) =
                        if method == 0x30 && pack_size_raw <= 16 {
                            (Rar3CheckMode::StoreCrc, pack_size_raw as u8)
                        } else {
                            (Rar3CheckMode::Heuristic, 0u8)
                        };

                    return Ok(rar3_info(Rar3Info {
                        salt,
                        enc_block,
                        check_mode,
                        head_type: 0,
                        file_crc,
                        pack_size,
                        auth_check: [0u8; 2],
                    }));
                }

                if next_off > data.len() { break; }
                off = next_off;
            }

            // ── End of archive ───────────────────────────────────
            0x7b => break,

            // ── Any other block (service, comment, AV, etc.) ─────
            _ => {
                if next_off > data.len() {
                    // Can't skip — block extends past EOF (e.g. large embedded
                    // comment). Scan forward for an encrypted file block.
                    if let Some(found) = scan_for_encrypted_file(data, off + 7) {
                        off = found;
                        continue;
                    }
                    break;
                }
                off = next_off;
            }
        }
    }

    // ── Last-resort: full forward scan ───────────────────────────
    // Handles malformed, multi-part, or unusual RAR3 archives where
    // normal block traversal fails to reach the encrypted file block.
    if let Some(found) = scan_for_encrypted_file(data, 7) {
        let head_size  = read_le16(data, found + 5) as usize;
        let head_flags = read_le16(data, found + 3);
        let has_salt   = head_flags & 0x0400 != 0;
        let next_off   = found + head_size;

        // Same pre-RAR-2.9 guard as the in-loop parser above.
        if found + 25 <= data.len() {
            let unp_ver = data[found + 24];
            if unp_ver < 29 {
                return parse_rar15_file_block(data, found, head_size, head_flags);
            }
        }

        let salt: [u8; 8] = if has_salt && head_size >= 8 && next_off <= data.len() {
            data[found + head_size - 8 .. found + head_size].try_into().unwrap()
        } else {
            archive_salt.unwrap_or([0u8; 8])
        };

        let enc_start = next_off.min(data.len());
        if enc_start + 16 > data.len() {
            bail!("RAR3: archive appears encrypted but data is truncated");
        }
        let enc_block: [u8; 16] = data[enc_start..enc_start+16].try_into().unwrap();

        let method        = if found + 26 <= data.len() { data[found + 25] } else { 0 };
        let file_crc      = if found + 20 <= data.len() {
            u32::from_le_bytes(data[found+16..found+20].try_into().unwrap())
        } else { 0 };
        let pack_size_raw = if found + 11 <= data.len() {
            u32::from_le_bytes(data[found+7..found+11].try_into().unwrap())
        } else { u32::MAX };

        let (check_mode, pack_size) =
            if method == 0x30 && pack_size_raw <= 16 {
                (Rar3CheckMode::StoreCrc, pack_size_raw as u8)
            } else {
                (Rar3CheckMode::Heuristic, 0u8)
            };

        return Ok(rar3_info(Rar3Info {
            salt,
            enc_block,
            check_mode,
            head_type: 0,
            file_crc,
            pack_size,
            auth_check: [0u8; 2],
        }));
    }

    bail!(
        "RAR3: no password-protected file found in archive \
         (the archive may not be encrypted, or uses an unsupported format)"
    );
}
