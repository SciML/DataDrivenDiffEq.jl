using DataDrivenDMD
using Aqua
using JET
using Test

@testset "QA" begin
    @testset "Aqua" begin
        Aqua.test_all(DataDrivenDMD)
    end
    @testset "JET" begin
        JET.test_package(DataDrivenDMD; target_defined_modules = true)
    end
end
