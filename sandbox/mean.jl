"""

TODOs:
- CompositeMean
- overload operators + and * for compising means
"""

export ZeroMean, CustomMean, ConstantMean


# Abstract base type
abstract type MeanFunction end


"""
    CustomMean <: MeanFunction

User-defined mean function, where `f(x)` should return a scalar for whatever
type of `x` this is intended to work with.
"""
struct CustomMean{T} <: MeanFunction
    f::T
end
@inline (f::CustomMean)(x) = f.f(x)


"""
    ZeroMean <: MeanFunction

Returns zero(s) everywhere.
"""
struct ZeroMean{T<:Real} <: MeanFunction end
@inline (::ZeroMean{T})(x) where T = zero(T)
@inline _map(z::ZeroMean{T}, D::AbstractVector) where T = Zeros{T}(length(D))
==(::ZeroMean, ::ZeroMean) = true


"""
    ConstantMean{T} <: MeanFunction

Returns `c` (of the appropriate type) everywhere.
"""
struct ConstantMean{T<:Real} <: MeanFunction
    c::T
end
@inline (mu::ConstantMean)(x) = mu.c
@inline _map(mu::ConstantMean, D::AbstractVector) = Fill(mu.c, length(D))
==(mu_a::ConstantMean, mu_b::ConstantMean) = mu_a.c == mu_b.c
+(mu_a::ConstantMean, mu_b::ConstantMean) = ConstantMean(mu_a.c + mu_b.c)
*(mu_a::ConstantMean, mu_b::ConstantMean) = ConstantMean(mu_a.c * mu_b.c)


################################################################################
# Mean utils
################################################################################

"""
    AbstractVector(mu::MeanFunction)

Convert `mu` into an `AbstractVector`, if such a representation exists.
"""
function AbstractVector(mu::MeanFunction)
    @assert isfinite(length(mu))
    return map(mu, eachindex(mu))
end
