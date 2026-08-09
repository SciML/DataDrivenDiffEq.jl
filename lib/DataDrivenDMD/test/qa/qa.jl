using SciMLTesting
using DataDrivenDMD
using JET

run_qa(
    DataDrivenDMD;
    reexports_allow = (:solve,)
)
