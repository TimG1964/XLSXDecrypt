module XLSXDecrypt

using Nettle   # used only for AES-CBC decryption
using SHA      # used for all hashing
using XML
using Base64

export decrypt_xlsx

# ─── PBKDF2-HMAC (manual, since Nettle.jl doesn't expose it) ─────────────────

function pbkdf2_hmac(hash_name::String, password::Vector{UInt8},
                     salt::Vector{UInt8}, iterations::Int, key_len::Int)
    dummy = HMACState(hash_name, password)
    update!(dummy, UInt8[0x00])
    hlen = length(digest!(dummy))

    nblocks = cld(key_len, hlen)
    dk = UInt8[]
    for i in 1:nblocks
        h = HMACState(hash_name, password)
        update!(h, salt)
        update!(h, UInt8[(i>>24)&0xff, (i>>16)&0xff, (i>>8)&0xff, i&0xff])  # big-endian
        u = digest!(h)
        t = copy(u)
        for _ in 2:iterations
            h = HMACState(hash_name, password)
            update!(h, u)
            u = digest!(h)
            t .⊻= u
        end
        append!(dk, t)
    end
    return dk[1:key_len]
end

# ─── CFB (OLE2 Compound File Binary) minimal parser ──────────────────────────

const CFB_MAGIC  = UInt8[0xD0,0xCF,0x11,0xE0,0xA1,0xB1,0x1A,0xE1]
const ENDOFCHAIN = UInt32(0xFFFFFFFE)   # end of a sector chain
const FREESECT   = UInt32(0xFFFFFFFF)   # unallocated sector
const FATSECT    = UInt32(0xFFFFFFFD)   # this sector IS a FAT sector
const DIFSECT    = UInt32(0xFFFFFFFC)   # this sector IS a DIFAT sector

# All special marker values are >= DIFSECT.
# IMPORTANT: arithmetic like FREESECT-2 wraps around on UInt32 and gives the
# wrong answer — always use this predicate instead of range comparisons.
is_special(s::UInt32) = s >= DIFSECT

function read_cfb_stream(io::IO, target_name::String)
    seekstart(io)
    read(io, 8) == CFB_MAGIC || error("Not a CFB/OLE2 file")

    # CFB header layout (byte offsets) per MS-CFB spec:
    #  0– 7  signature / magic (8 bytes)         ← already read
    #  8–23  CLSID (16 bytes, unused)
    # 24–25  minor version
    # 26–27  major version
    # 28–29  byte order mark (0xFFFE)
    # 30–31  sector size (power of 2)
    # 32–33  mini sector size (power of 2)
    # 34–39  reserved (6 bytes)
    # 40–43  num directory sectors (v4 only; 0 for v3)
    # 44–47  num FAT sectors
    # 48–51  first directory sector
    # 52–55  transaction signature
    # 56–59  mini stream cutoff size
    # 60–63  first mini-FAT sector
    # 64–67  num mini-FAT sectors
    # 68–71  first DIFAT sector
    # 72–75  num DIFAT sectors
    # 76–    inline DIFAT table (109 × 4 bytes = 436 bytes)
    seek(io, 30)
    sector_size      = 1 << read(io, UInt16)    # 30–31: sector size power (512 → 9)
    mini_sector_size = 1 << read(io, UInt16)    # 32–33: mini sector size power (64 → 6)
    seek(io, 44)
    num_fat          = read(io, UInt32)          # 44–47: number of FAT sectors
    first_dir_sector = read(io, UInt32)          # 48–51: first directory sector
    read(io, UInt32)                             # 52–55: transaction signature (skip)
    mini_cutoff      = read(io, UInt32)          # 56–59: mini stream cutoff (usually 4096)
    first_minifat    = read(io, UInt32)          # 60–63: first mini-FAT sector
    num_minifat      = read(io, UInt32)          # 64–67: number of mini-FAT sectors
    first_difat      = read(io, UInt32)          # 68–71: first DIFAT sector
    num_difat        = read(io, UInt32)          # 72–75: number of DIFAT sectors

    # ── Collect FAT sector locations from the inline DIFAT table (109 entries)
    difat = UInt32[]
    for _ in 1:109
        e = read(io, UInt32)
        !is_special(e) && push!(difat, e)
    end

    # ── Walk any extra DIFAT sectors (only for files with >109 FAT sectors — rare)
    sec = first_difat
    for _ in 1:num_difat
        is_special(sec) && break
        seek(io, (sec + 1) * sector_size)
        for _ in 1:(sector_size ÷ 4 - 1)
            e = read(io, UInt32)
            !is_special(e) && push!(difat, e)
        end
        sec = read(io, UInt32)
    end

    # ── Build the FAT — read exactly num_fat sectors
    fat = UInt32[]
    for s in difat[1:min(num_fat, length(difat))]
        seek(io, (s + 1) * sector_size)
        for _ in 1:(sector_size ÷ 4)
            push!(fat, read(io, UInt32))
        end
    end

    # Follow a FAT chain one step. Returns ENDOFCHAIN sentinel on any problem.
    # This is intentionally lenient: 0, out-of-bounds, and special values all
    # terminate the chain, because some CFB writers use 0 instead of ENDOFCHAIN.
    function next_fat(s::UInt32)::UInt32
        is_special(s) && return ENDOFCHAIN
        idx = Int(s) + 1
        idx > length(fat) && return ENDOFCHAIN
        nxt = fat[idx]
        # Treat 0 as end-of-chain: some writers zero-fill unused FAT entries
        nxt == 0x00000000 && return ENDOFCHAIN
        nxt
    end

    # ── Build the mini-FAT (walk exactly num_minifat sectors)
    mini_fat = UInt32[]
    if num_minifat > 0 && !is_special(first_minifat)
        sec = first_minifat
        for _ in 1:num_minifat
            is_special(sec) && break
            seek(io, (sec + 1) * sector_size)
            for _ in 1:(sector_size ÷ 4)
                push!(mini_fat, read(io, UInt32))
            end
            sec = next_fat(sec)
        end
    end

    # ── Read directory entries (128 bytes each).
    # Upper bound: a typical CFB has very few directory sectors (often just 1).
    # We cap at 8192 entries (1 MB of directory data) to prevent runaway reads.
    dir_entries = NamedTuple[]
    seen_dir    = Set{UInt32}()
    sec         = first_dir_sector
    max_dir_sectors = max(1, cld(8192 * 128, sector_size))
    for _ in 1:max_dir_sectors
        is_special(sec) && break
        sec in seen_dir && break        # genuine cycle — stop, don't error
        push!(seen_dir, sec)
        seek(io, (sec + 1) * sector_size)
        for _ in 1:(sector_size ÷ 128)
            raw_name  = read(io, 64)
            name_len  = read(io, UInt16)
            obj_type  = read(io, UInt8)
            read(io, 1)          # color flag
            read(io, 4)          # left-sibling SID
            read(io, 4)          # right-sibling SID
            read(io, 4)          # child SID
            read(io, 36)         # CLSID + state + timestamps
            start_sec = read(io, UInt32)
            stream_sz = read(io, UInt64)

            entry_name = ""
            if name_len >= 2
                nchars = (name_len - 2) ÷ 2
                chars  = reinterpret(UInt16, raw_name[1:2*nchars])
                entry_name = String(Char.(chars))
            end
            push!(dir_entries, (name=entry_name, type=obj_type,
                                 start=start_sec, size=stream_sz))
        end
        sec = next_fat(sec)
    end

    isempty(dir_entries) && error("No directory entries found in CFB file")

    # ── Entry 0 is always root storage; its sector chain holds the mini-stream
    root_entry = dir_entries[1]

    mini_container = UInt8[]
    seen_mc = Set{UInt32}()
    sec = root_entry.start
    while !is_special(sec) && !(sec in seen_mc)
        push!(seen_mc, sec)
        seek(io, (sec + 1) * sector_size)
        append!(mini_container, read(io, sector_size))
        sec = next_fat(sec)
    end

    # ── Find the requested stream (directory entry type 0x02 = stream object)
    entry = nothing
    for e in dir_entries
        if e.name == target_name && e.type == 0x02
            entry = e; break
        end
    end
    entry === nothing && error("Stream '$target_name' not found in CFB file")

    # ── Read the stream data.
    # `remaining` is the authoritative termination condition — we stop as soon
    # as we have the declared number of bytes, regardless of FAT chain values.
    data      = UInt8[]
    remaining = Int(entry.size)

    if entry.size < mini_cutoff
        # Small stream: data lives in the mini-stream
        sec = entry.start
        while remaining > 0
            is_special(sec) && error("mini-stream chain ended early for '$target_name'")
            off   = Int(sec) * mini_sector_size
            chunk = min(mini_sector_size, remaining)
            off + chunk > length(mini_container) && error("mini-stream out of bounds for '$target_name'")
            append!(data, mini_container[off+1 : off+chunk])
            remaining -= chunk
            remaining == 0 && break
            idx = Int(sec) + 1
            idx > length(mini_fat) && error("mini-FAT chain ended early for '$target_name'")
            sec = mini_fat[idx]
        end
    else
        # Large stream: data lives in normal FAT sectors
        sec = entry.start
        while remaining > 0
            is_special(sec) && error("FAT chain ended early for '$target_name'")
            seek(io, (sec + 1) * sector_size)
            chunk = min(sector_size, remaining)
            append!(data, read(io, chunk))
            remaining -= chunk
            remaining == 0 && break
            sec = next_fat(sec)
        end
    end

    return data
end

# ─── XML.jl helper: depth-first search for first element with a given tag ────

# Match on local name only (strips namespace prefix like "p:" or "enc:")
local_tag(node) = let t = XML.tag(node); something(findfirst(':', t), 0) == 0 ? t : t[findfirst(':', t)+1:end] end

function find_node(node, target_tag::String)
    XML.nodetype(node) == XML.Element || return nothing
    local_tag(node) == target_tag && return node
    for child in XML.children(node)
        XML.nodetype(child) == XML.Element || continue
        result = find_node(child, target_tag)
        result !== nothing && return result
    end
    return nothing
end

# ─── ECMA-376 Agile Encryption: parse EncryptionInfo XML ─────────────────────

function parse_encryption_info(info_bytes::Vector{UInt8})
    # Bytes 1–8 are a version/reserved header; XML starts at byte 9
    xml_str = String(info_bytes[9:end])
    doc     = XML.parse(XML.Node, xml_str)

    # XML.parse may return a Document node; unwrap to the first Element child
    root = if XML.nodetype(doc) == XML.Element
        doc
    else
        first(c for c in XML.children(doc) if XML.nodetype(c) == XML.Element)
    end

    kd = find_node(root, "keyData")
    ek = find_node(root, "encryptedKey")

    kd === nothing && error("Could not find <keyData> in EncryptionInfo")
    ek === nothing && error("Could not find <encryptedKey> in EncryptionInfo")

    ka = XML.attributes(kd)
    ea = XML.attributes(ek)

    return (
        # keyData attributes (used for final package decryption)
        cipher_alg      = ka["cipherAlgorithm"],
        cipher_chaining = ka["cipherChaining"],
        hash_alg        = ka["hashAlgorithm"],
        key_bits        = parse(Int, ka["keyBits"]),
        block_size      = parse(Int, ka["blockSize"]),
        salt_size       = parse(Int, ka["saltSize"]),
        key_data_salt   = base64decode(ka["saltValue"]),

        # encryptedKey attributes (used for key unwrapping + password verification)
        spin_count      = parse(Int, ea["spinCount"]),
        enc_key_salt    = base64decode(ea["saltValue"]),
        enc_salt_size   = parse(Int, ea["saltSize"]),
        enc_block_size  = parse(Int, ea["blockSize"]),
        enc_hash_alg    = ea["hashAlgorithm"],
        enc_key_bits    = parse(Int, ea["keyBits"]),
        enc_verifier_hash_input = base64decode(ea["encryptedVerifierHashInput"]),
        enc_verifier_hash_value = base64decode(ea["encryptedVerifierHashValue"]),
        enc_key_value           = base64decode(ea["encryptedKeyValue"]),
    )
end

# ─── ECMA-376 §2.3.4.11  key derivation ──────────────────────────────────────

const HASH_NAME_MAP = Dict(
    "SHA512" => SHA.sha512,
    "SHA256" => SHA.sha256,
    "SHA1"   => SHA.sha1,
    "SHA384" => SHA.sha384,
)

# Convert a UInt32 to 4 little-endian bytes
function uint32le(x::UInt32)::Vector{UInt8}
    return UInt8[(x >> 0) & 0xff, (x >> 8) & 0xff, (x >> 16) & 0xff, (x >> 24) & 0xff]
end

function get_hash_fn(alg::String)
    fn = get(HASH_NAME_MAP, alg, nothing)
    fn === nothing && error("Unsupported hash algorithm: $alg")
    fn
end

function derive_key(password::String, salt::Vector{UInt8}, spin_count::Int,
                    hash_alg::String, key_bits::Int, block_size::Int,
                    block_key::Vector{UInt8})
    hash_fn = get_hash_fn(hash_alg)

    # UTF-16LE encode the password explicitly
    utf16_units = transcode(UInt16, password)
    pwd_utf16 = Vector{UInt8}(undef, length(utf16_units) * 2)
    for (i, u) in enumerate(utf16_units)
        pwd_utf16[2i-1] = u & 0xff
        pwd_utf16[2i]   = (u >> 8) & 0xff
    end

    # Step 1: H(salt || UTF-16LE(password))
    h_bytes = hash_fn([salt; pwd_utf16])

    # Step 2: iterate spin_count times: H(LE32(i) || h_bytes)
    for i in 0:(spin_count - 1)
        h_bytes = hash_fn([uint32le(UInt32(i)); h_bytes])
    end

    # Step 3: H(h_bytes || block_key), truncate/pad to key_bits÷8 bytes
    dk = hash_fn([h_bytes; block_key])

    key_len = key_bits ÷ 8
    if length(dk) < key_len
        append!(dk, fill(0x36, key_len - length(dk)))
    end
    return dk[1:key_len]
end


# ─── AES-CBC decrypt via Nettle ───────────────────────────────────────────────

function aes_cbc_decrypt(key::Vector{UInt8}, iv::Vector{UInt8},
                          ciphertext::Vector{UInt8})
    cipher_name = "AES$(length(key) * 8)"
    dec = Decryptor(cipher_name, key)
    return decrypt(dec, :CBC, iv, ciphertext)
end

# ─── Main public function ─────────────────────────────────────────────────────

"""
    decrypt_xlsx(path::String, password::String) -> IOBuffer

Decrypts a password-protected .xlsx file and returns an `IOBuffer` containing
the plaintext .xlsx data. Pass the result directly to `XLSX.readxlsx` or
`XLSX.openxlsx`.

```julia
buf = decrypt_xlsx("protected.xlsx", "secret")
xf  = XLSX.readxlsx(buf)
```

Only the modern ECMA-376 Agile Encryption scheme (Excel 2010+) is supported.
"""
function decrypt_xlsx(path::String, password::String)::IOBuffer
    raw     = read(path)
    file_io = IOBuffer(raw)

    # 1. Extract the two CFB streams
    enc_info_bytes = read_cfb_stream(file_io, "EncryptionInfo")
    enc_pkg_bytes  = read_cfb_stream(file_io, "EncryptedPackage")

    # 2. Parse encryption parameters from the XML inside EncryptionInfo
    p = parse_encryption_info(enc_info_bytes)

    # ECMA-376 §2.3.4.11 Table 1 — fixed block key constants
    BLOCK_VERIFIER_INPUT = UInt8[0xfe,0xa7,0xd2,0x76,0x3b,0x4b,0x9e,0x79]
    BLOCK_VERIFIER_HASH  = UInt8[0xd7,0xaa,0x0f,0x6d,0x30,0x61,0x34,0x4e]
    BLOCK_KEY_VALUE      = UInt8[0x14,0x6e,0x0b,0xe7,0xab,0xac,0xd0,0xd6]

    # 3. Derive three intermediate keys from password + encryptedKey salt.
    #    Must use enc_block_size (from encryptedKey) not block_size (from keyData).
    key_vi = derive_key(password, p.enc_key_salt, p.spin_count,
                        p.enc_hash_alg, p.enc_key_bits, p.enc_block_size,
                        BLOCK_VERIFIER_INPUT)
    key_vh = derive_key(password, p.enc_key_salt, p.spin_count,
                        p.enc_hash_alg, p.enc_key_bits, p.enc_block_size,
                        BLOCK_VERIFIER_HASH)
    key_kv = derive_key(password, p.enc_key_salt, p.spin_count,
                        p.enc_hash_alg, p.enc_key_bits, p.enc_block_size,
                        BLOCK_KEY_VALUE)

    # 4. Verify the password.
    dec_vi = aes_cbc_decrypt(key_vi, p.enc_key_salt, p.enc_verifier_hash_input)
    dec_vh = aes_cbc_decrypt(key_vh, p.enc_key_salt, p.enc_verifier_hash_value)

    computed = get_hash_fn(p.enc_hash_alg)(dec_vi[1:p.enc_salt_size])

    computed == dec_vh[1:length(computed)] ||
        error("Wrong password (verifier mismatch)")

    # 5. Decrypt the actual encryption key
    actual_key = aes_cbc_decrypt(key_kv, p.enc_key_salt, p.enc_key_value)
    actual_key = actual_key[1:(p.key_bits ÷ 8)]

    # 6. Decrypt EncryptedPackage
    #    First 8 bytes = uint64LE giving the true plaintext size
    plaintext_size = only(reinterpret(UInt64, enc_pkg_bytes[1:8]))
    ciphertext     = enc_pkg_bytes[9:end]

    seg_size       = 4096
    n_segments     = cld(length(ciphertext), seg_size)
    plaintext      = UInt8[]
    sizehint!(plaintext, length(ciphertext))

    for i in 0:(n_segments - 1)
        # Per-segment IV = H(keyDataSalt || LE32(i))[1:block_size]
        seg_iv = get_hash_fn(p.hash_alg)([p.key_data_salt; uint32le(UInt32(i))])[1:p.block_size]

        seg_start  = i * seg_size + 1
        seg_end    = min((i + 1) * seg_size, length(ciphertext))
        seg_cipher = ciphertext[seg_start:seg_end]

        # Pad to AES block boundary if needed
        rem = mod(length(seg_cipher), p.block_size)
        rem != 0 && append!(seg_cipher, zeros(UInt8, p.block_size - rem))

        append!(plaintext, aes_cbc_decrypt(actual_key, seg_iv, seg_cipher))
    end

    return IOBuffer(plaintext[1:plaintext_size])
end

end # module
