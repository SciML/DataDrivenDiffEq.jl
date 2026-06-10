using DataDrivenLux
using Aqua
using JET
using Test

@testset "QA" begin
    @testset "Aqua" begin
        Aqua.test_all(DataDrivenLux)
    end
    @testset "JET" begin
        JET.test_package(DataDrivenLux; target_defined_modules = true)
    end
end
