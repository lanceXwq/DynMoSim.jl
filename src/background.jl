"""
    FieldOfView(lo, hi)

An axis-aligned imaging volume, from corner `lo` to corner `hi`, two 3-vectors. For a 2D dataset it is still a volume: [`project`](@ref) drops z, and the projection of a point uniform in the volume is uniform in its xy rectangle.

Used to place spurious localizations (see [`Background`](@ref)). It does **not** clip signal localizations — a structure may sit partly or wholly outside the field of view and will still be imaged in full.
"""
struct FieldOfView{T}
    lo::SVector{3,T}
    hi::SVector{3,T}

    # An inner constructor, so that passing SVectors directly cannot slip past the check — Julia would otherwise synthesise an exact-match outer constructor and validation would apply only to the conversion path.
    function FieldOfView{T}(lo::SVector{3,T}, hi::SVector{3,T}) where {T}
        all(hi .> lo) ||
            throw(ArgumentError("every hi must exceed its lo, got lo = $lo, hi = $hi"))
        return new{T}(lo, hi)
    end
end

FieldOfView(lo::SVector{3,T}, hi::SVector{3,T}) where {T} = FieldOfView{T}(lo, hi)

function FieldOfView(lo::AbstractVector, hi::AbstractVector)
    length(lo) == length(hi) == 3 || throw(
        DimensionMismatch(
            "lo and hi must be 3-vectors, got lengths $(length(lo)) and $(length(hi))"
        ),
    )
    return FieldOfView{Float64}(SVector{3,Float64}(lo), SVector{3,Float64}(hi))
end

"""
    randin([rng], f::FieldOfView) -> SVector

A point uniform in the field of view.
"""
@inline randin(rng::AbstractRNG, f::FieldOfView{T}) where {T} =
    f.lo .+ (f.hi .- f.lo) .* rand(rng, SVector{3,T})

randin(f::FieldOfView) = randin(default_rng(), f)

# ---------------------------------------------------------------------------

"""
    Background(; rate, fov, meanphotons)

Spurious localizations — the false positives a real fitting pipeline produces from noise, autofluorescence and out-of-focus emitters.

Background events arrive as their own Poisson process at `rate` events per unit time, independent of the structure, and are placed uniformly in `fov`.

`meanphotons` is their own photon budget, separate from the signal's: spurious detections are typically dimmer than real fluorophores, and therefore also less precisely localized, since both share the same PSF and uncertainty model. Pass `meanphotons(optics)` to make them indistinguishable in brightness.

Because two independent Poisson processes superpose into one, `simulate` draws a single stream at `λ_signal + rate` and labels each event signal with probability `λ_signal / (λ_signal + rate)`. That is exact and costs one exponential draw per event rather than two processes and a merge.
"""
struct Background{T}
    rate::T
    fov::FieldOfView{T}
    meanphotons::T
end

function Background(; rate, fov::FieldOfView{T}, meanphotons) where {T}
    rate >= 0 || throw(ArgumentError("background rate must be non-negative, got $rate"))
    meanphotons > 0 ||
        throw(ArgumentError("background meanphotons must be positive, got $meanphotons"))
    return Background{T}(T(rate), fov, T(meanphotons))
end

"""
    meanphotons(bg::Background)

Mean photons of a spurious event, including the ones that emit nothing and are dropped. Mirrors [`meanphotons(::Optics)`](@ref).
"""
meanphotons(bg::Background) = bg.meanphotons

"""
    photondist(bg::Background) -> Geometric

Photon count of a spurious event, `Geometric(1 / (1 + meanphotons(bg)))` — the same family as the signal's, with its own mean. Zeros are dropped exactly as they are for signal events.
"""
photondist(bg::Background) = Geometric(1 / (1 + bg.meanphotons))

@inline _psignal(λs::Real, λb::Real) = λs / (λs + λb)
