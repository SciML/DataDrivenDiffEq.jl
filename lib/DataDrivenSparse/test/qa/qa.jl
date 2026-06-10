using DataDrivenSparse
using Aqua
using JET
using Test

@testset "QA" begin
    @testset "Aqua" begin
        Aqua.test_all(DataDrivenSparse)
    end
    @testset "JET" begin
        JET.test_package(DataDrivenSparse; target_defined_modules = true)
    end
end
