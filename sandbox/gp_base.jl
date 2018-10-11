""" Gaussian process base module.

TODOs:
- Abstract base class?

"""

import Statistics: mean, cov
import Distributions: logpdf, ContinuousMultivariateDistribution

export GP, kernel, mean_vec, marginals, ==
# rand, logpdf, elbo, marginal_cov, marginal_std, xcov, GPC


"""
    GP{mu<:MeanFunction, k<:Kernel}

A Gaussian Process (GP) object. Either constructed using an Affine Transformation of
existing GPs or by providing a mean function `μ`, a kernel `k`, and a `GPC` `gpc`.
-----

Note ContinuousMultivariateDistribution := Distribution{Multivariate, Continuous}
"""
struct GaussianProcess{mu<:MeanFunction, k<:Kernel} <: ContinuousMultivariateDistribution
    # args::Any
    mu_::mu
    k_::k
end

const GP = GaussianProcess

"""
    GP{mu::Real, k<:Kernel}

GP initialized with a constant mean, e.g. the zero mean.
"""
function GP(m::Real, k::Kernel)
    if isfinite(length(k))
        if iszero(m)
            return GP(zero(EmpiricalMean(Zeros(length(k)))), k, gpc)  # hack
        else
            return GP(FiniteMean(ConstantMean(m), eachindex(k)), k, gpc)
        end
    else
        if iszero(m)
            return GP(zero(ConstantMean(m)), k, gpc)  # hack
        else
            return GP(ConstantMean(m), k, gpc)
        end
    end
end
GP(k::Kernel) = GP(ZeroMean{Float64}(), k)
GP() = GP(ZeroMean{Float64}(), RBFKernel())


"""
"""
mean(f::GP) = f.mu_  # okay overloading the Statistics `mean`??


"""
"""
kernel(f::GP) = f.k_
# """
#     kernel(f::Union{Real, Function})
#     kernel(f::AbstractGP)
#     kernel(f::Union{Real, Function}, g::AbstractGP)
#     kernel(f::AbstractGP, g::Union{Real, Function})
#     kernel(fa::AbstractGP, fb::AbstractGP)

# Get the cross-kernel between `GP`s `fa` and `fb`, and . If either argument is deterministic
# then the zero-kernel is returned. Also, `kernel(f) === kernel(f, f)`.
# """
# kernel(f::GP) = f.k
# function kernel(fa::GP, fb::GP)
#     @assert fa.gpc === fb.gpc
#     if fa === fb
#         return kernel(fa)
#     elseif fa.args == nothing && fa.n > fb.n || fb.args == nothing && fb.n > fa.n
#         if isfinite(length(fa)) && isfinite(length(fb))
#             return FiniteZeroCrossKernel(eachindex(fa), eachindex(fb))
#         elseif isfinite(length(fa)) && !isfinite(length(fb))
#             return LhsFiniteZeroCrossKernel(eachindex(fa))
#         elseif !isfinite(length(fa)) && isfinite(length(fb))
#             return RhsFiniteZeroCrossKernel(eachindex(fb))
#         else # Both processes are infinite dimensional.
#             return ZeroKernel{Float64}()
#         end
#     elseif fa.n > fb.n
#         return k_p′p(fa.args..., fb)
#     else
#         return k_pp′(fa, fb.args...)
#     end
# end
# kernel(::Union{Real, Function}) = ZeroKernel{Float64}()
# kernel(::Union{Real, Function}, ::GP) = ZeroKernel{Float64}()
# kernel(::GP, ::Union{Real, Function}) = ZeroKernel{Float64}()






"""
    mean_vec(f::GP)

The `Vector` representation of the mean function of `f`, if `isfinite(length(f))`.
"""
mean_vec(f::GP) = AbstractVector(mean(f))

"""
    cov(f::GP)

The covariance matrix of `f`, if `isfinite(length(f))`.
"""
cov(f::GP) = AbstractMatrix(kernel(f))







################################################################################
# Helpers and sugars
################################################################################

"""
    marginals(f::GP)

Sugar, equivalent to `(mean(f), marginal_std(f))`.
"""
marginals(f::GP) = (mean_vec(f), marginal_std(f))

"""
    length(f::GP)

Dimensionality of the space over which `f` distributes. Usuall infinite b/c
the GP distributes over a function, but may be finite for the case of
multivariate normal.
"""
length(f::GP) = length(mean(f))

"""
    eachindex(f::GP)

Enumerate the dimensions of the space over which `f` distributes (if such an
enumeration exists and is finite).
"""
eachindex(f::GP) = eachindex(mean(f))

"""
    ==(f::GP, g::GP)

Two GPs `f` and `g` are equal iff their marginal distributions are equal -- i.e.,
the mean function and kernels of both are equal.
"""
==(f::GP, g::GP) = (mean(f) == mean(g)) && (kernel(f) == kernel(g))


