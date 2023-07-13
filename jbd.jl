module JBD

using DifferentialEquations
using Printf

struct Parameters
    ω::Float64    # Brans-Dicke coupling
    ϕini::Float64 # initial value of scalar field (at early time)
    Ωr0::Float64  # radiation   density parameter
    Ωm0::Float64  # matter      density parameter
    ΩΛ0::Float64  # dark energy density parameter
    # (Hubble parameter H0 is not needed for integrating scalar field)
end

# find two (real) roots (x1, x2) of quadratic equation a*x^2 + b*x + c*x = 0
function quadroots(a, b, c)
    d = b^2 - 4*a*c
    x1 = d >= 0.0 ? (-b + √(d)) / (2*a) : NaN
    x2 = d >= 0.0 ? (-b - √(d)) / (2*a) : NaN
    return x1, x2
end

# ODE system to solve (as function of x = loga):
#   y1  = logϕ,
#   y2  = a^3 * E * ϕ * dlogϕ/dloga,
#   y1′ = dlogϕ / dloga,
#   y2′ = d(a^3*E*ϕ*dlogϕ/dloga) / dloga = 3 * a^3 * (Ωm0/a^3 + 4*ΩΛ0) / ((3 + 2*ω) * E),
# where E(t)^2 = (H/H0)^2 = 1/ϕ * (Ωr0/a^4 + Ωm0/a^3 + ΩΛ0) / (1 + dlogϕ/dloga - ω/6 * (dlogϕ/dloga)^2)

a(x) = exp(x) # x = log(a)

       logϕ(θ::Parameters, y, x) = y[1]
          ϕ(θ::Parameters, y, x) = exp(logϕ(θ, y, x))
dlogϕ_dloga(θ::Parameters, y, x) = quadroots(a(x)^6 * ϕ(θ,y,x) * (θ.Ωr0/a(x)^4 + θ.Ωm0/a(x)^3 + θ.ΩΛ0) + θ.ω/6 * y[2]^2, -y[2]^2, -y[2]^2)[1] # positive solution y1′=dlogϕ/dloga of quadratic equation y1′ = y2/(a^3*E(y1′)*ϕ) with E(y1′) expanded
          E(θ::Parameters, y, x) = √( 1/ϕ(θ, y, x) * (θ.Ωr0/a(x)^4 + θ.Ωm0/a(x)^3 + θ.ΩΛ0) / (1 + dlogϕ_dloga(θ, y, x) - θ.ω/6 * (dlogϕ_dloga(θ, y, x))^2) )

function integrate_ϕ(θ::Parameters; aini=1e-10)
    y1′(y, x) = dlogϕ_dloga(θ, y, x)
    y2′(y, x) = 3 * a(x)^3 / (3+2*θ.ω) * (θ.Ωm0/a(x)^3 + 4*θ.ΩΛ0) / E(θ, y, x)
    y′(y, _, x) = [y1′(y, x), y2′(y, x)]
    y0 = [log(θ.ϕini), 0.0] # assumption: dlogϕ_dloga = 0 (so y2′ = 0) at early a = aini
    x0 = log(aini)
    return solve(ODEProblem(y′, y0, (x0, 0.0)), abstol=1e-10, reltol=1e-10)
end

# find JBD cosmology with ϕini and ΩΛ0 such that ϕ0 = 1 and Ω0 = 1 by (fixed-point) iteration
# TODO: varying G/G != 1?
function shoot(ω=1e3, Ωr0=5.5e-5, Ωm0=0.317; maxiters=100, verbose=false)
    println("Fixing ω = $ω, Ωr0 = $Ωr0, Ωm0 = $Ωm0;")
    println("Varying ΩΛ0, ϕini until ϕ0 = 1 and Ω0 = 1:")

    ϕini_lo = 0.0 # lower bound, guaranteed to have ϕ0 < 1.0 (since it gives ϕ0 = 0)
    ϕini_hi = 1.0 # upper bound, guaranteed to have ϕ0 > 1.0 (since dlogϕ/dloga > 0)

    ΩΛ0 = 1.0 - Ωr0 - Ωm0 # initial *guess* for ΩΛ0 (assuming ΩΦ0 = 0); will be refined by the loop

    for iter in 1:maxiters
        ϕini = (ϕini_lo + ϕini_hi) / 2

        # construct cosmology with current iteration of parameters
        θ = Parameters(ω, ϕini, Ωr0, Ωm0, ΩΛ0)

        # integrate cosmology
        y = integrate_ϕ(θ) # callable ODE solution y(x) = [y(x)[1], y(x)[2]]
        x0  = 0.0
        y0  = y(x0)
        ϕ0  = ϕ(θ, y0, x0)
        Ωϕ0 = -dlogϕ_dloga(θ, y0, x0) + θ.ω/6 * dlogϕ_dloga(θ, y0, x0)^2 # defined so that we should have Ω0 = ∑(Ωi0) = 1 (from E0 = 1)
        Ω0  = θ.Ωr0 + θ.Ωm0 + θ.ΩΛ0 + Ωϕ0

        # print information about cosmology
        if verbose
            @printf "#%02d: JBD with ϕini = %.10f, ΩΛ0 = %.10f gives ϕ0 = %.10f, Ω0 = %.10f\n" iter ϕini θ.ΩΛ0 ϕ0 Ω0
        end

        # have we found the desired cosmology with ϕ0 = 1 and Ω0 = 1?
        if ϕini_lo ≈ ϕini_hi && ϕ0 ≈ 1.0 && Ω0 ≈ 1.0
            return θ # converged within maxiters
        end

        # if not, refine the guesses
        # * for ΩΛ0 based on the closure condition ∑(Ωi0) = 1
        # * for Φini based on whether ϕ0 hit above/below target value
        # and reiterate (the scalar field integral changes when ΩΛ0 and/or ϕini changes)
        ΩΛ0 = 1.0 - Ωr0 - Ωm0 - Ωϕ0 # closure condition
        if ϕ0 > 1.0        # hit above,
            ϕini_hi = ϕini # so need lower  ϕini
        else               # hit below,
            ϕini_lo = ϕini # so need higher ϕini
        end
    end

    return nothing # did not converge within maxiters
end

θ = shoot(verbose=true) # the desired cosmology

end
