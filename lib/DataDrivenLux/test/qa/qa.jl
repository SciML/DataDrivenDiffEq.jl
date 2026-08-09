using SciMLTesting
using DataDrivenLux
using JET

# AbstractDifferentiation.gradient and Optim.converged are documented owner APIs,
# but their packages do not yet declare these bindings public.
run_qa(
    DataDrivenLux;
    ei_kwargs = (;
        all_qualified_accesses_are_public = (; ignore = (:gradient, :converged)),
    )
)
