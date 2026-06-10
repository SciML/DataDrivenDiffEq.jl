using DataDrivenSR
using Aqua
using JET
using Test

@testset "QA" begin
    @testset "Aqua" begin
        Aqua.test_all(DataDrivenSR)
    end
    @testset "JET" begin
        JET.test_package(DataDrivenSR; target_defined_modules = true)
    end
end
