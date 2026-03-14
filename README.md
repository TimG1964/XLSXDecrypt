
# XLSXDecrypt.jl

An experimental package to allow XLSX.jl to open password encrypted Excel workbooks.

This package was written by Claude with a small amount of input from me. I do not understand encryption. My simple test case works but it has not been extensively tested.

## Requirements

* Julia v1.9 or higher

* XLSX.jl v0.10 or higher

* Linux, macOS or Windows.

## Installation

From a Julia session, run:

```julia
julia> using Pkg

julia> Pkg.add("XLSXDecrypt")
```

## Usage

This package offers only one public function:

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

Only the modern ECMA-376 Agile Encryption scheme (Excel 2010+) is supported.

## Source Code

The source code for this package is hosted at
[https://github.com/TimG1964/XLSXDecrypt.jl](https://github.com/TimG1964/XLSXDecrypt.jl).

## License

The source code for the package **XLSXDecrypt.jl** is licensed under
the [MIT License](https://raw.githubusercontent.com/TimG1964/XLSXDecrypt.jl/master/LICENSE).

## Getting Help

If you're having any trouble, have any questions about this package
or want to ask for a new feature,
just open a new [issue](https://github.com/TimG1964/XLSXDecrypt.jl/issues).

## Contributing

Contributions are always welcome!

To contribute, fork the project from [GitHub](https://github.com/TimG1964/XLSXDecrypt.jl)
and send a Pull Request.
