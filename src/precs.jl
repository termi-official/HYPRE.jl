struct BoomerAMGPrecWrapper{MatType}
    P::HYPRE.BoomerAMG
    A::MatType
end

function LinearAlgebra.ldiv!(y::AbstractVector, prec::BoomerAMGPrecWrapper, x::AbstractVector)
    fill!(y, eltype(y)(0.0))
    HYPRE.solve!(prec.P, y, prec.A, x)
end

"""
    BoomerAMGPrecBuilder(settings_fun; kwargs...)
"""
struct BoomerAMGPrecBuilder{SFun, Tk}
    settings_fun!::SFun
    kwargs::Tk
end

# Syntactic sugar wth some defaults
function BoomerAMGPrecBuilder(settings_fun! = (amg, A, p) -> nothing; kwargs...)
    return BoomerAMGPrecBuilder(settings_fun!, kwargs)
end

function (b::BoomerAMGPrecBuilder)(A, p)
    amg = HYPRE.BoomerAMG(; b.kwargs...)
    Internals.set_precond_defaults(amg)
    b.settings_fun!(amg, A, p)
    return (BoomerAMGPrecWrapper(amg, A), LinearAlgebra.I)
end
