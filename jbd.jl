module JBD

using DifferentialEquations
using Printf

struct Parameters
    ω::Float64    # Brans-Dicke coupling
    ϕini::Float64 # initial value of scalar field at early times
    ωr0::Float64  # physical radiation   density parameter (= Ωr0*h0^2)
    ωm0::Float64  # physical matter      density parameter (= Ωm0*h0^2)
    ωΛ0::Float64  # physical dark energy density parameter (= ΩΛ0*h0^2)
end

# find two roots (x1, x2) of quadratic equation a*x^2 + b*x + c*x = 0
function quadroots(a, b, c)
    d = b^2 - 4*a*c
    x1 = d >= 0.0 ? (-b + √(d)) / (2*a) : NaN
    x2 = d >= 0.0 ? (-b - √(d)) / (2*a) : NaN
    return x1, x2
end

a(x) = exp(x) # x = log(a)

# ODE system to solve (as function of x = loga):
#   y1  = logϕ,
#   y2  = a^3 * h * ϕ * dlogϕ/dloga,
#   y1′ = dlogϕ / dloga,
#   y2′ = d(a^3*h*ϕ*dlogϕ/dloga) / dloga = 3 * a^3 * (ωm0/a^3 + 4*ωΛ0) / ((3 + 2ω) * h),
# where h(t)^2 = (H(t) / (100 km/(s*Mpc)))^2 = 1/ϕ * (ωr0/a^4 + ωm0/a^3 + ωΛ0) / (1 + dlogϕ/dloga - ω/6 * (dlogϕ/dloga)^2)
# is *time-dependent* reduced Hubble parameter (to make notation consistent)

       logϕ(θ::Parameters, y, x) = y[1]
          ϕ(θ::Parameters, y, x) = exp(logϕ(θ, y, x))
dlogϕ_dloga(θ::Parameters, y, x) = quadroots(a(x)^6 * ϕ(θ,y,x) * (θ.ωr0/a(x)^4 + θ.ωm0/a(x)^3 + θ.ωΛ0) + θ.ω/6 * y[2]^2, -y[2]^2, -y[2]^2)[1] # positive solution y1′=dlogϕ/dloga of quadratic equation y1′ = y2/(a^3*h(y1′)*ϕ) with h(y1′) expanded
          h(θ::Parameters, y, x) = √( 1/ϕ(θ, y, x) * (θ.ωr0/a(x)^4 + θ.ωm0/a(x)^3 + θ.ωΛ0) / (1 + dlogϕ_dloga(θ, y, x) - θ.ω/6 * (dlogϕ_dloga(θ, y, x))^2) )

function integrate_ϕ(θ::Parameters; aini=1e-10)
    y1′(y, x) = dlogϕ_dloga(θ, y, x)
    y2′(y, x) = 3 * a(x)^3 / (3+2*θ.ω) * (θ.ωm0/a(x)^3 + 4*θ.ωΛ0) / h(θ, y, x)
    y′(y, _, x) = [y1′(y, x), y2′(y, x)]
    y0 = [log(θ.ϕini), 0.0] # assumption: dlogϕ_dloga = 0 (so y2′ = 0) at early a = aini
    x0 = log(aini)
    return solve(ODEProblem(y′, y0, (x0, 0.0)), abstol=1e-6, reltol=1e-6)
end

# find ϕini and ωΛ0 such that ϕ0 = 1 and Ω0 = 1
function shoot(ω=1e3, ωr0=5.5e-5*0.67^2, ωm0=0.317*0.67^2; maxiters=100, verbose=false)
    println("Fixing ω = $ω, ωr0 = $ωr0, ωm0 = $ωm0;")
    println("Varying ωΛ0, ϕini until ϕ0 = 1 and Ω0 = 1:")

    ϕini_lo = 0.0 # lower bound, guaranteed to have ϕ0 < 1.0 (since it gives ϕ0 = 0)
    ϕini_hi = 1.0 # upper bound, guaranteed to have ϕ0 > 1.0 (since dlogϕ/dloga > 0)

    ωΛ0 = 0.0 # extremely stupid initial *guess*; will be refined by the loop

    for iter in 1:maxiters
        ϕini = (ϕini_lo + ϕini_hi) / 2

        # integrate scalar field with current cosmological parameters
        θ = Parameters(ω, ϕini, ωr0, ωm0, ωΛ0)
        y = integrate_ϕ(θ) # callable ODE solution y(x) = [y(x)[1], y(x)[2]]

        # present values of various quantities in the found cosmology
        x0  = 0.0
        y0  = y(x0)
        ϕ0  = ϕ(θ, y0, x0)
        h0  = h(θ, y0, x0) # *derived* Hubble parameter
        Ωr0 = θ.ωr0/h0^2
        Ωm0 = θ.ωm0/h0^2
        ΩΛ0 = θ.ωΛ0/h0^2
        Ωϕ0 = -dlogϕ_dloga(θ, y0, x0) + θ.ω/6 * dlogϕ_dloga(θ, y0, x0)^2 # defined so that we should have Ω0 = ∑(Ωi0) = 1
        Ω0  = Ωr0 + Ωm0 + ΩΛ0 + Ωϕ0 # *should* be 1 when we have found the right value for ωΛ0

        # print information about this cosmology
        if verbose
            @printf "#%02d: JBD with ϕini = %.10f, ωΛ0 = %.10f has ϕ0 = %.10f, h0 = %.10f, ΩΛ0 = %.10f Ω0 = %.10f\n" iter ϕini θ.ωΛ0 ϕ0 h0 ΩΛ0 Ω0
        end

        # we want the cosmology with (ϕini, ωΛ0) such that ϕ0 = 1 and Ω0 = 1
        # (the latter usually follows from the former, but we check to be sure)
        if ϕini_lo ≈ ϕini_hi && ϕ0 ≈ 1.0 && Ω0 ≈ 1.0
            return θ # converged within maxiters
        end

        # update ωΛ0 to the value that satisfies the closure condition ∑(Ωi0) = 1 for the h0 in this cosmology
        # (then we must reiterate to correct the scalar field integral, which changes then ωΛ0 changes)
        ΩΛ0 = 1 - Ωr0 - Ωm0 - Ωϕ0 # closure condition; equivalent to E = 1
        ωΛ0 = ΩΛ0*h0^2

        # update ϕ0 (adjust aim), then shoot again
        if ϕ0 > 1.0  # hit above,
            ϕini_hi = ϕini # so need lower  ϕini
        else               # hit below,
            ϕini_lo = ϕini # so need higher ϕini
        end
    end

    return nothing # did not converge within maxiters
end

θ = shoot(verbose=true) # the desired cosmology

end
