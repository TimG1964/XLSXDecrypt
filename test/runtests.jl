import XLSX
import XLSXDecrypt as XD
using Test
using Dates


data_directory = joinpath(dirname(pathof(XD)), "..", "data")

@assert isdir(data_directory)

@testset "Basic functionality" begin
    test_file = joinpath(data_directory, raw"password-is-w23$er3.xlsx")
    io=XD.decrypt_xlsx(test_file, raw"w23$er3")
    test_file2 = joinpath(data_directory, raw"password-is-very$long^password#3245301!.xlsx")
    io2=XD.decrypt_xlsx(test_file2, raw"very$long^password#3245301!")

    @testset "number formats" begin
        XLSX.openxlsx(io) do f
            show(IOBuffer(), f)
            sheet = f["Sheet1"]
            @test sheet["A1"] == 1
            @test isapprox(sheet["B1"], 0.546832775750823)
            @test sheet["C1"] == "kjghfjvila"
            @test sheet["A2"] == 2
            @test isapprox(sheet["B2"], 0.381845788463574)
            @test sheet["C2"] == "ghfjkqwefg"
            @test isapprox(sheet["B3"], 0.541686223027816)

            @test sheet["A5"] == 10
            @test isapprox(sheet["B5"], 1.78799032829419)
            @test isapprox(f["Sheet2!B3"], 0.541686223027816)
            @test f["Sheet2!C3"] == "fhlAWETYUUI"
            @test isapprox(f["Sheet2!B4"], 0.317625541051977)
            @test f["Sheet2!C4"] == "HFJuwe"
        end
        XLSX.openxlsx(io2) do f
            show(IOBuffer(), f)
            sheet = f["Sheet1"]
            @test sheet["A1"] == 1
            @test isapprox(sheet["B1"], 0.546832775750823)
            @test sheet["C1"] == "kjghfjvila"
            @test sheet["A2"] == 2
            @test isapprox(sheet["B2"], 0.381845788463574)
            @test sheet["C2"] == "ghfjkqwefg"
            @test isapprox(sheet["B3"], 0.541686223027816)

            @test sheet["A5"] == 10
            @test isapprox(sheet["B5"], 1.78799032829419)
            @test isapprox(f["Sheet2!B3"], 0.541686223027816)
            @test f["Sheet2!C3"] == "fhlAWETYUUI"
            @test isapprox(f["Sheet2!B4"], 0.317625541051977)
            @test f["Sheet2!C4"] == "HFJuwe"
        end
    end

    @testset "Defined Names" begin

        seekstart(io) 
        f = XLSX.openxlsx(io, mode="rw")
        s = f["Sheet2"]
        @test all(isapprox.(s["Floats"], [0.546832775750823; 0.381845788463574; 0.541686223027816; 0.317625541051977;;]))
        @test s["SortedStrings"] == Any["fhlAWETYUUI"; "ghfjkqwefg"; "HFJuwe"; "kjghfjvila";;]

        seekstart(io2) 
        f = XLSX.openxlsx(io2, mode="rw")
        s = f["Sheet2"]
        @test all(isapprox.(s["Floats"], [0.546832775750823; 0.381845788463574; 0.541686223027816; 0.317625541051977;;]))
        @test s["SortedStrings"] == Any["fhlAWETYUUI"; "ghfjkqwefg"; "HFJuwe"; "kjghfjvila";;]

    end
end

v= pkgversion(XLSX)
if (v.major, v.minor) >= (0, 11)

    @testset "Newer functionality" begin

        test_file = joinpath(data_directory, raw"password-is-w23$er3.xlsx")
        io=XD.decrypt_xlsx(test_file, raw"w23$er3")
        f = XLSX.openxlsx(io, mode="rw")
        test_file = joinpath(data_directory, raw"password-is-very$long^password#3245301!.xlsx")
        io=XD.decrypt_xlsx(test_file, raw"very$long^password#3245301!")
        f2 = XLSX.openxlsx(io, mode="rw")

        @testset "formulas" begin
            s = f[1]
            wb = XLSX.get_workbook(s)
            @test XLSX.getcell(s, "A5") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("A5"), "", "13", "10", "", true)
            @test XLSX.get_formula_from_cache(s, XLSX.CellRef("A5")) == XLSX.Formula("SUM(A1:A4)", nothing, nothing, nothing)
            @test XLSX.getcell(s, "D1") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("D1"), "s", "6", "2", "1", true)
            @test XLSX.get_formula_from_cache(s, XLSX.CellRef("D1")) == XLSX.Formula("_xlfn._xlws.SORT(C1:C4)", "array", "D1:D4", nothing)

            s = f2[1]
            wb = XLSX.get_workbook(s)
            wb = XLSX.get_workbook(s)
            @test XLSX.getcell(s, "A5") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("A5"), "", "13", "10", "", true)
            @test XLSX.get_formula_from_cache(s, XLSX.CellRef("A5")) == XLSX.Formula("SUM(A1:A4)", nothing, nothing, nothing)
            @test XLSX.getcell(s, "D1") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("D1"), "s", "6", "2", "1", true)
            @test XLSX.get_formula_from_cache(s, XLSX.CellRef("D1")) == XLSX.Formula("_xlfn._xlws.SORT(C1:C4)", "array", "D1:D4", nothing)
        end

        @testset "formatting" begin

            s = f[1]
            @test XLSX.getFont(s, "A1").font == Dict("name" => Dict("val" => "Aptos Narrow"), "family" => Dict("val" => "2"), "sz" => Dict("val" => "11"), "color" => Dict("rgb" => "FF006100"), "scheme" => Dict("val" => "minor"))
            @test XLSX.getFont(s, "B2").font == Dict("name" => Dict("val" => "Aptos Narrow"), "family" => Dict("val" => "2"), "sz" => Dict("val" => "11"), "color" => Dict("rgb" => "FF9C5700"), "scheme" => Dict("val" => "minor"))
            @test XLSX.getFont(s, "C3").font == Dict("name" => Dict("val" => "Aptos Narrow"), "family" => Dict("val" => "2"), "sz" => Dict("val" => "11"), "color" => Dict("rgb" => "FF9C0006"), "scheme" => Dict("val" => "minor"))
            @test XLSX.getFont(s, "D4").font == Dict("name" => Dict("val" => "Aptos Narrow"), "family" => Dict("val" => "2"), "sz" => Dict("val" => "11"), "color" => Dict("theme" => "1"), "scheme" => Dict("val" => "minor"))
            @test XLSX.getFill(s, "D2").fill == Dict("patternFill" => Dict("patternType" => "solid", "fgrgb" => "FFFFFFCC"))
            @test XLSX.getFill(s, "C3").fill == Dict("patternFill" => Dict("patternType" => "solid", "fgrgb" => "FFFFC7CE"))
            @test XLSX.getFill(s, "B4").fill == Dict("patternFill" => Dict("patternType" => "solid", "fgrgb" => "FFFFEB9C"))
            @test XLSX.getFill(s, "A5").fill == Dict("patternFill" => Dict("patternType" => "solid", "fgrgb" => "FFF2F2F2"))
            @test XLSX.getBorder(s, "A5").border == Dict("left" => Dict("rgb" => "FF7F7F7F", "style" => "thick"), "bottom" => Dict("rgb" => "FF7F7F7F", "style" => "thick"), "right" => Dict("rgb" => "FF7F7F7F", "style" => "thin"), "top" => Dict("rgb" => "FF7F7F7F", "style" => "double"), "diagonal" => nothing)
            @test XLSX.getBorder(s, "B5").border == Dict("left" => Dict("rgb" => "FF7F7F7F", "style" => "thin"), "bottom" => Dict("rgb" => "FF7F7F7F", "style" => "thick"), "right" => Dict("rgb" => "FF7F7F7F", "style" => "thick"), "top" => Dict("rgb" => "FF7F7F7F", "style" => "double"), "diagonal" => nothing)
            @test XLSX.getBorder(s, "C2").border == Dict("left" => Dict("indexed" => "64", "style" => "thin"), "bottom" => Dict("indexed" => "64", "style" => "thin"), "right" => Dict("indexed" => "64", "style" => "thin"), "top" => Dict("indexed" => "64", "style" => "thin"), "diagonal" => nothing)
            @test XLSX.getBorder(s, "D1").border == Dict("left" => Dict("indexed" => "64", "style" => "thin"), "bottom" => Dict("indexed" => "64", "style" => "thin"), "right" => Dict("indexed" => "64", "style" => "medium"), "top" => Dict("indexed" => "64", "style" => "medium"), "diagonal" => nothing)

            s = f2[1]
            @test XLSX.getFont(s, "A1").font == Dict("name" => Dict("val" => "Aptos Narrow"), "family" => Dict("val" => "2"), "sz" => Dict("val" => "11"), "color" => Dict("rgb" => "FF006100"), "scheme" => Dict("val" => "minor"))
            @test XLSX.getFont(s, "B2").font == Dict("name" => Dict("val" => "Aptos Narrow"), "family" => Dict("val" => "2"), "sz" => Dict("val" => "11"), "color" => Dict("rgb" => "FF9C5700"), "scheme" => Dict("val" => "minor"))
            @test XLSX.getFont(s, "C3").font == Dict("name" => Dict("val" => "Aptos Narrow"), "family" => Dict("val" => "2"), "sz" => Dict("val" => "11"), "color" => Dict("rgb" => "FF9C0006"), "scheme" => Dict("val" => "minor"))
            @test XLSX.getFont(s, "D4").font == Dict("name" => Dict("val" => "Aptos Narrow"), "family" => Dict("val" => "2"), "sz" => Dict("val" => "11"), "color" => Dict("theme" => "1"), "scheme" => Dict("val" => "minor"))
            @test XLSX.getFill(s, "D2").fill == Dict("patternFill" => Dict("patternType" => "solid", "fgrgb" => "FFFFFFCC"))
            @test XLSX.getFill(s, "C3").fill == Dict("patternFill" => Dict("patternType" => "solid", "fgrgb" => "FFFFC7CE"))
            @test XLSX.getFill(s, "B4").fill == Dict("patternFill" => Dict("patternType" => "solid", "fgrgb" => "FFFFEB9C"))
            @test XLSX.getFill(s, "A5").fill == Dict("patternFill" => Dict("patternType" => "solid", "fgrgb" => "FFF2F2F2"))
            @test XLSX.getBorder(s, "A5").border == Dict("left" => Dict("rgb" => "FF7F7F7F", "style" => "thick"), "bottom" => Dict("rgb" => "FF7F7F7F", "style" => "thick"), "right" => Dict("rgb" => "FF7F7F7F", "style" => "thin"), "top" => Dict("rgb" => "FF7F7F7F", "style" => "double"), "diagonal" => nothing)
            @test XLSX.getBorder(s, "B5").border == Dict("left" => Dict("rgb" => "FF7F7F7F", "style" => "thin"), "bottom" => Dict("rgb" => "FF7F7F7F", "style" => "thick"), "right" => Dict("rgb" => "FF7F7F7F", "style" => "thick"), "top" => Dict("rgb" => "FF7F7F7F", "style" => "double"), "diagonal" => nothing)
            @test XLSX.getBorder(s, "C2").border == Dict("left" => Dict("indexed" => "64", "style" => "thin"), "bottom" => Dict("indexed" => "64", "style" => "thin"), "right" => Dict("indexed" => "64", "style" => "thin"), "top" => Dict("indexed" => "64", "style" => "thin"), "diagonal" => nothing)
            @test XLSX.getBorder(s, "D1").border == Dict("left" => Dict("indexed" => "64", "style" => "thin"), "bottom" => Dict("indexed" => "64", "style" => "thin"), "right" => Dict("indexed" => "64", "style" => "medium"), "top" => Dict("indexed" => "64", "style" => "medium"), "diagonal" => nothing)

        end
    end
end

