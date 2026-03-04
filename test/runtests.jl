import XLSX
import XLSXDecrypt as XD
using Test
using Dates


data_directory = joinpath(dirname(pathof(XD)), "..", "data")

@assert isdir(data_directory)

@testset "Simple decrypt" begin
    test_file = joinpath(data_directory, raw"password-is-w23$er3.xlsx")
    io=XD.decrypt_xlsx(test_file, raw"w23$er3")

    @testset "number formats" begin
        XLSX.openxlsx(io) do f
            show(IOBuffer(), f)
            sheet = f["general"]
            @test sheet["A1"] == "text"
            @test sheet["B1"] == "regular text"
            @test sheet["A2"] == "integer"
            @test sheet["B2"] == 102
            @test sheet["A3"] == "float"
            @test isapprox(sheet["B3"], 102.2)
            @test sheet["A4"] == "date"
            @test sheet["B4"] == Date(1983, 4, 16)
            @test sheet["A5"] == "hour"
            @test sheet["B5"] == Dates.Time(Dates.Hour(19), Dates.Minute(45))
            @test sheet["A6"] == "datetime"
            @test sheet["B6"] == Date(2018, 4, 16) + Dates.Time(Dates.Hour(19), Dates.Minute(19), Dates.Second(51))
            @test f["general!B7"] == -220.0
            @test f["general!B8"] == -2000
            @test f["general!B9"] == 100000000000000
            @test f["general!B10"] == -100000000000000
        end
    end

    @testset "Defined Names" begin # Issue #148
        seekstart(io) 
        f = XLSX.opentemplate(io)
        @test f["SINGLE_CELL"] == "single cell A2"
        @test f["RANGE_B4C5"] == Any["range B4:C5" "range B4:C5"; "range B4:C5" "range B4:C5"]
        @test f["CONST_DATE"] == 43383
        @test isapprox(f["CONST_FLOAT"], 10.2)
        @test f["CONST_INT"] == 100
        @test f["LOCAL_INT"] == 2000
        @test f["named_ranges_2"]["LOCAL_INT"] == 2000
        @test f["named_ranges"]["LOCAL_INT"] == 1000
        @test f["named_ranges"]["LOCAL_NAME"] == "Hey You"
        @test f["named_ranges_2"]["LOCAL_NAME"] == "out there in the cold"
        @test f["named_ranges"]["SINGLE_CELL"] == "single cell A2"

        @test_throws XLSX.XLSXError f["header_error"]["LOCAL_REF"]
        @test f["named_ranges"]["LOCAL_REF"][1] == 10
        @test f["named_ranges"]["LOCAL_REF"][2] == 20
        @test f["named_ranges_2"]["LOCAL_REF"][1] == "local"
        @test f["named_ranges_2"]["LOCAL_REF"][2] == "reference"

        XLSX.addDefinedName(f["lookup"], "Life_the_Universe_and_Everything", 42)
        XLSX.addDefinedName(f["lookup"], "FirstName", "Hello World")
        XLSX.addDefinedName(f["lookup"], "single", "C2"; absolute=true)
        XLSX.addDefinedName(f["lookup"], "range", "C3:C5"; absolute=true)
        XLSX.addDefinedName(f["lookup"], "NonContig", "C3:C5,D3:D5"; absolute=true)
        @test f["lookup"]["Life_the_Universe_and_Everything"] == 42
        @test f["lookup"]["FirstName"] == "Hello World"
        @test f["lookup"]["single"] == "NAME"
        @test f["lookup"]["range"] == Any["name1"; "name2"; "name3";;] # A 2D Array, size (3, 1)
        @test f["lookup"]["NonContig"] == [["name1"; "name2"; "name3";;], [100; 200; 300;;]] # NonContiguousRanges return a vector of matrices

        XLSX.addDefinedName(f, "Life_the_Universe_and_Everything", 42)
        XLSX.addDefinedName(f, "FirstName", "Hello World")
        XLSX.addDefinedName(f, "single", "lookup!C2"; absolute=true)
        XLSX.addDefinedName(f, "range", "lookup!C3:C5"; absolute=true)
        XLSX.addDefinedName(f, "NonContig", "lookup!C3:C5,lookup!D3:D5"; absolute=true)
        @test f["Life_the_Universe_and_Everything"] == 42
        @test f["FirstName"] == "Hello World"
        @test f["single"] == "NAME"
        @test f["range"] == Any["name1"; "name2"; "name3";;] # A 2D Array, size (3, 1)
        @test f["NonContig"] == [["name1"; "name2"; "name3";;], [100; 200; 300;;]] # NonContiguousRanges return a vector of matrices

        XLSX.setFont(f["lookup"], "NonContig"; name="Arial", size=12, color="FF0000FF", bold=true, italic=true, under="single", strike=true)
        @test XLSX.getFont(f["lookup"], "C3").font == Dict("i" => nothing, "b" => nothing, "u" => nothing, "strike" => nothing, "sz" => Dict("val" => "12"), "name" => Dict("val" => "Arial"), "color" => Dict("rgb" => "FF0000FF"))
        @test XLSX.getFont(f["lookup"], "C4").font == Dict("i" => nothing, "b" => nothing, "u" => nothing, "strike" => nothing, "sz" => Dict("val" => "12"), "name" => Dict("val" => "Arial"), "color" => Dict("rgb" => "FF0000FF"))
        @test XLSX.getFont(f["lookup"], "C5").font == Dict("i" => nothing, "b" => nothing, "u" => nothing, "strike" => nothing, "sz" => Dict("val" => "12"), "name" => Dict("val" => "Arial"), "color" => Dict("rgb" => "FF0000FF"))
        @test XLSX.getFont(f["lookup"], "D3").font == Dict("i" => nothing, "b" => nothing, "u" => nothing, "strike" => nothing, "sz" => Dict("val" => "12"), "name" => Dict("val" => "Arial"), "color" => Dict("rgb" => "FF0000FF"))
        @test XLSX.getFont(f["lookup"], "D4").font == Dict("i" => nothing, "b" => nothing, "u" => nothing, "strike" => nothing, "sz" => Dict("val" => "12"), "name" => Dict("val" => "Arial"), "color" => Dict("rgb" => "FF0000FF"))
        @test XLSX.getFont(f["lookup"], "D5").font == Dict("i" => nothing, "b" => nothing, "u" => nothing, "strike" => nothing, "sz" => Dict("val" => "12"), "name" => Dict("val" => "Arial"), "color" => Dict("rgb" => "FF0000FF"))
        XLSX.setFont(f, "single"; name="Arial", size=12, color="FF0000FF", bold=true, italic=true, under="double", strike=true)
        @test XLSX.getFont(f["lookup"], "C2").font == Dict("i" => nothing, "b" => nothing, "u" => Dict("val" => "double"), "strike" => nothing, "sz" => Dict("val" => "12"), "name" => Dict("val" => "Arial"), "color" => Dict("rgb" => "FF0000FF"))

        XLSX.writexlsx("mytest.xlsx", f, overwrite=true)

        f = XLSX.readxlsx("mytest.xlsx")
        @test f["Life_the_Universe_and_Everything"] == 42
        @test f["FirstName"] == "Hello World"
        @test f["single"] == "NAME"
        @test f["range"] == Any["name1"; "name2"; "name3";;] # A 2D Array, size (3, 1)
        @test f["NonContig"] == [["name1"; "name2"; "name3";;], [100; 200; 300;;]] # NonContiguousRanges return a vector of matrices
        isfile("mytest.xlsx") && rm("mytest.xlsx")


    end

    @testset "ReferencedFormulae" begin

        test_file = joinpath(data_directory, raw"password-is-very$long^password#3245301!.xlsx")
        io=XD.decrypt_xlsx(test_file, raw"very$long^password#3245301!")
        f = XLSX.openxlsx(io, mode="rw")

        s = f[1]
        wb = XLSX.get_workbook(s)
        @test XLSX.getcell(s, "A2") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("A2"), "", "", "20", "", true)
        @test XLSX.get_formula_from_cache(s, XLSX.CellRef("A2")) == XLSX.ReferencedFormula("SUM(O2:S2)", 0, "A2:A10", nothing)
        @test XLSX.getcell(s, "A3") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("A3"), "", "", "25", "", true)
        @test XLSX.get_formula_from_cache(s, XLSX.CellRef("A3")) == XLSX.FormulaReference(0, nothing)
        s["A2"] = 3
        @test XLSX.getcell(s, "A2") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("A2"), "", "", "3", "", false)
        @test XLSX.getcell(s, "A3") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("A3"), "", "", "25", "", true)
        @test XLSX.get_formula_from_cache(s, XLSX.CellRef("A3")) == XLSX.ReferencedFormula("SUM(O3:S3)", 0, "A3:A10", nothing)
        @test XLSX.get_formula_from_cache(s, XLSX.CellRef("A4")) == XLSX.FormulaReference(0, nothing)
        
        s2 = f[2]
        @test XLSX.getcell(s2, "A1") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("A1"), "", "", "1", "", true)
        @test XLSX.get_formula_from_cache(s2, XLSX.CellRef("A1")) == XLSX.Formula("SECOND(NOW())", nothing, nothing, Dict("ca" => "1"))
        @test XLSX.getcell(s2, "A2") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("A2"), "", "", "1", "", true)
        @test XLSX.get_formula_from_cache(s2, XLSX.CellRef("A2")) == XLSX.ReferencedFormula("SECOND(NOW())", 1, "A2:A5", Dict("ca" => "1"))
        s2["A2"] = 3
        @test XLSX.getcell(s2, "A2") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("A2"), "", "", "3", "", false)
        @test XLSX.get_formula_from_cache(s2, XLSX.CellRef("A3")).id == 1
        @test XLSX.get_formula_from_cache(s2, XLSX.CellRef("A3")).unhandled == Dict("ca" => "1")
        @test XLSX.getcell(s2, "A3").formula == true
        @test XLSX.getcell(s2, "A3") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("A3"), "", "", "1", "", true)
        @test XLSX.get_formula_from_cache(s2, XLSX.CellRef("A3")) == XLSX.ReferencedFormula("SECOND(NOW())", 1, "A3:A5", Dict("ca" => "1"))
        @test XLSX.getcell(s2, "B1") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("B1"), "", "", "1", "", true)
        @test XLSX.get_formula_from_cache(s2, XLSX.CellRef("B1")) == XLSX.ReferencedFormula("SECOND(NOW())", 0, "B1:C5", Dict("ca" => "1"))
        s2["B1"] = 3
        @test XLSX.getcell(s2, "B1") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("B1"), "", "", "3", "", false)
        @test XLSX.getcell(s2, "B2") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("B2"), "", "", "1", "", true)
        @test XLSX.get_formula_from_cache(s2, XLSX.CellRef("B2")) == XLSX.ReferencedFormula("SECOND(NOW())", 0, "B2:C5", Dict("ca" => "1"))
        @test XLSX.getcell(s2, "C1") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("C1"), "", "", "1", "", true)
        @test XLSX.get_formula_from_cache(s2, XLSX.CellRef("C1")) == XLSX.Formula("SECOND(NOW())", nothing, "C1", Dict("ca" => "1"))

        XLSX.writexlsx("mytest.xlsx", f, overwrite=true)
        f2 = XLSX.openxlsx("mytest.xlsx", mode="rw")

        s = f2[1]
        @test XLSX.getcell(s, "A2") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("A2"), "", "", "3", "", false)
        @test XLSX.getcell(s, "A3") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("A3"), "", "", "25", "", true)
        @test XLSX.get_formula_from_cache(s, XLSX.CellRef("A3")) == XLSX.ReferencedFormula("SUM(O3:S3)", 0, "A3:A10", nothing)

        s2 = f[2]
        @test XLSX.getcell(s2, "A2") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("A2"), "", "", "3", "", false)
        @test XLSX.get_formula_from_cache(s2, XLSX.CellRef("A3")).id == 1
        @test XLSX.get_formula_from_cache(s2, XLSX.CellRef("A3")).unhandled == Dict("ca" => "1")
        @test XLSX.getcell(s2, "A3") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("A3"), "", "", "1", "", true)
        @test XLSX.get_formula_from_cache(s2, XLSX.CellRef("A3")) == XLSX.ReferencedFormula("SECOND(NOW())", 1, "A3:A5", Dict("ca" => "1"))
        @test XLSX.getcell(s2, "B1") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("B1"), "", "", "3", "", false)
        @test XLSX.getcell(s2, "B2") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("B2"), "", "", "1", "", true)
        @test XLSX.get_formula_from_cache(s2, XLSX.CellRef("B2")) == XLSX.ReferencedFormula("SECOND(NOW())", 0, "B2:C5", Dict("ca" => "1"))
        @test XLSX.getcell(s2, "C1") == XLSX.Cell(XLSX.get_workbook(f), XLSX.CellRef("C1"), "", "", "1", "", true)
        @test XLSX.get_formula_from_cache(s2, XLSX.CellRef("C1")) ==XLSX.Formula("SECOND(NOW())", nothing, "C1", Dict("ca" => "1"))

        isfile("mytest.xlsx") && rm("mytest.xlsx")
    end
end
