"""
    AbstractShape

A reference configuration to draw label sites from. Subtypes implement

    sample_site(rng, shape) -> SVector{3,T}

One site is drawn per emission event, so `sample_site` is the primitive and `sample_sites` is a convenience wrapper over it. Shapes are conventionally defined at unit size so that a deformation's `scale` sets the physical dimensions.

Sampling a fresh site per event makes labels a continuous density over the structure rather than a fixed set of discrete points.

Shapes are 3D, like the deformations that move them; a 2D dataset is the projection of a 3D one (see [`project`](@ref)).
"""
abstract type AbstractShape end

"""
    sample_site([rng], shape) -> SVector

Draw one material coordinate from `shape`. This is the primitive each shape implements; it allocates nothing.
"""
function sample_site end

sample_site(shape::AbstractShape) = sample_site(default_rng(), shape)

"""
    sample_sites([rng], shape, n) -> Vector{SVector}

Draw `n` material coordinates from `shape`.
"""
function sample_sites(rng::AbstractRNG, shape::AbstractShape, n::Integer)
    return [sample_site(rng, shape) for _ in 1:n]
end

sample_sites(shape::AbstractShape, n::Integer) = sample_sites(default_rng(), shape, n)

# ---------------------------------------------------------------------------

"""
    SphereSurface()

Uniform points on the unit sphere.
"""
struct SphereSurface <: AbstractShape end

@inline sample_site(rng::AbstractRNG, ::SphereSurface) =
    normalize(randn(rng, SVector{3,Float64}))

"""
    Ball()

Uniform points inside the unit ball.
"""
struct Ball <: AbstractShape end

@inline sample_site(rng::AbstractRNG, ::Ball) =
    normalize(randn(rng, SVector{3,Float64})) * rand(rng)^(1 / 3)

"""
    Ring()

Uniform points on the unit circle in the xy-plane (`z = 0`). A `Similarity`'s `rot` tilts it.
"""
struct Ring <: AbstractShape end

@inline function sample_site(rng::AbstractRNG, ::Ring)
    φ = 2π * rand(rng)
    return SVector(cos(φ), sin(φ), 0.0)
end

"""
    LineSegment(a, b)

Uniform points on the segment from `a` to `b`, two 3-vectors.
"""
struct LineSegment{T} <: AbstractShape
    a::SVector{3,T}
    b::SVector{3,T}
end

function LineSegment(a::AbstractVector, b::AbstractVector)
    length(a) == length(b) == 3 || throw(
        DimensionMismatch(
            "LineSegment needs two 3-vectors, got lengths $(length(a)) and $(length(b))"
        ),
    )
    return LineSegment(SVector{3,Float64}(a), SVector{3,Float64}(b))
end

@inline sample_site(rng::AbstractRNG, s::LineSegment) = s.a .+ rand(rng) .* (s.b .- s.a)

"""
    PointCloud(pts)

An explicit reference configuration: wrap a `3 × N` matrix or a vector of 3-vectors. Sampling draws one of the points uniformly at random, so a point cloud behaves like any other shape — use it when you have model points rather than a parametric form.

With a fresh draw per event, the same point can be selected repeatedly, which is the discrete analogue of a label emitting more than once.
"""
struct PointCloud{T} <: AbstractShape
    pts::Vector{SVector{3,T}}
end

PointCloud(pts::AbstractMatrix{<:Real}) = PointCloud(tosites(pts))

@inline sample_site(rng::AbstractRNG, s::PointCloud) =
    @inbounds s.pts[rand(rng, eachindex(s.pts))]

# ---------------------------------------------------------------------------

"""
    tosites(pts) -> Vector{SVector{3,T}}

Convert a `3 × N` matrix of points into static vectors. Accepts a vector of 3-vectors unchanged, so callers can take either form.
"""
function tosites(pts::AbstractMatrix{T}) where {T<:Real}
    size(pts, 1) == 3 || throw(DimensionMismatch("expected 3 rows, got $(size(pts, 1))"))
    return [SVector{3,float(T)}(view(pts, :, j)) for j in axes(pts, 2)]
end

tosites(pts::AbstractVector{<:SVector{3}}) = collect(pts)
