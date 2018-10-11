

# Abstract base type
abstract type Kernel end

# Some ABC-like defs
@inline isstationary(::Type{<:Kernel}) = false
@inline isstationary(x::Kernel) = isstationary(typeof(x))
size(k::Kernel) = (length(k), length(k))
size(k::Kernel, dim::Int) = size(k)[dim]
length(k::Kernel) = Inf
# eachindex(k::Kernel, N::Int) = eachindex(k)



"""
    EQ <: Kernel

The standardised Exponentiated Quadratic kernel (with no free parameters?).
"""
struct EQ{len<:Real} <: Kernel
    len_::len
end
EQ() = EQ(0.5)
isstationary(::Type{<:EQ}) = true
(::EQ)(x1, x2) = exp(-0.5 * sqeuclidean(x1, x2))
(::EQ)(x1::Real, x2::Real) = exp(-0.5 * (x1 - x2)^2)
# (::EQ)(x::T) where T = one(Float64)
_pairwise(::EQ, X::ColsAreObs) = exp.(-0.5 .* pairwise(SqEuclidean(), X.X))
_pairwise(::EQ, X::ColsAreObs, X′::ColsAreObs) = exp.(-0.5 .* pairwise(SqEuclidean(), X.X, X′.X))
@inline eachindex(k::EQ) = eachindex_err(k)


"""
    PerEQ{Tp<:Real}

The usual periodic kernel derived by mapping the input domain onto a circle.
"""
struct PerEQ{Tp<:Real} <: Kernel
    p::Tp
end
PerEQ() = PerEQ(1.0)
isstationary(::Type{<:PerEQ}) = true
(k::PerEQ)(x1::Real, x2::Real) = exp(-2 * sin(pi * abs(x1 - x2) / k.p)^2)  # l=1
(k::PerEQ)(x::Real) = one(typeof(x))
# @inline eachindex(k::PerEQ) = eachindex_err(k)


"""
    Noise{T<:Real} <: Kernel

A white-noise kernel with a single scalar parameter.
"""
struct Noise{T<:Real} <: Kernel
    var::T
end
isstationary(::Type{<:Noise}) = true
==(a::Noise, b::Noise) = a.var == b.var
(k::Noise)(x_a, x_b) =
    # x_a === x_b || x_a == x_b ? k.var : zero(k.var)
    x_a == x_b ? k.var : zero(k.var)
(k::Noise)(x) = k.var
_pairwise(k::Noise, X::AV) = Diagonal(Fill(k.var, length(X)))
function _pairwise(k::Noise, X_a::AV, X_b::AV)
    if X_a === X_b
        return _pairwise(k, X_a)
    else
        return [view(X_a, p) == view(X_b, q) ? k.var : 0
            for p in eachindex(X_a), q in eachindex(X_b)]
    end
end
@inline eachindex(k::Noise) = eachindex_err(k)


################################################################################
# Kernel utils
################################################################################

"""
    AbstractMatrix(k::Kernel)

Convert `k` into an `AbstractMatrix`, if such a representation exists.
"""
function AbstractMatrix(k::Kernel)
    @assert isfinite(size(k, 1))
    return pairwise(k, eachindex(k, 1))
end

