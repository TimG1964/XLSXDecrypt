
# XLSXDecrypt.jl

Experimental package to allow XLSX.jl to open password encrypted workbooks.

This package was written by Claude with a small amount of input from me. I do not understand encryption. My simple test case works but it has not been extensively tested.

It offers only one public function:

```julia
    decrypt_xlsx(filename::String, password::String)
```

This returns an `IOBuffer` that either `XLSX.readxlsx` or `XLSX.openxlsx` can ingest.

Thus:

```julia
julia> using XLSXDecrypt, XLSX

julia> buf = decrypt_xlsx("password.xlsx", "password")

julia> f=openxlsx(buf, mode="rw")
XLSXFile("IOBuffer(data=UInt8[...], readable=true, writable=false, seekable=true, append=false, size=8554, maxsize=Inf, ptr=8555, mark=-1)") containing 1 Worksheet
            sheetname size          range
-------------------------------------------------
               Sheet1 3x1           A1:A3
```