using SciMLTesting
using DataDrivenDiffEq

run_qa(
    DataDrivenDiffEq;
    reexports_allow = (:solve,)
)
