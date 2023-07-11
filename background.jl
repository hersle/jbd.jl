module JBD

using DifferentialEquations
using Printf

struct Parameters
    ω::Float64   # Brans-Dicke coupling
    ωr0::Float64 # physical radiation   density parameter (= Ωr0*h0^2)
    ωm0::Float64 # physical matter      density parameter (= Ωm0*h0^2)
    ωΛ0::Float64 # physical dark energy density parameter (= ΩΛ0*h0^2)
end

function fixed_point_iterate(f::Function, x0; maxiters=100)
    for iter in 1:maxiters
        x = f(x0) # update next value
        if x ≈ x0
            return x # converged within maxiters
        end
        x0 = x # update previous value
    end
    return NaN # did not converge within maxiters
end

a(x) = exp(x) # x = log(a)

# ODE system to solve:
#     y[1] = logϕ
#     y[2] = a^3 * h * ϕ * dlogϕ/dloga
#     dy_dx[1] = dlogϕ / dloga
#     dy_dx[2] = d(a^3*h*ϕ*dlogϕ/dloga) / dloga = 3 * a^3 * (ωm0/a^3 + 4*ωΛ0) / ((3 + 2ω) * h)
# (h(t) = H(t) / (100 km/(s*Mpc)) is *time-dependent* reduced Hubble parameter, to make notation consistent)
function integrate_ϕ(θ::Parameters, ϕini; aini=1e-10)
    dy_dx(y, _, x) = [dlogϕ_dloga(θ, y, x), 3 / (3+2*θ.ω) * a(x)^3 * (θ.ωm0/a(x)^3 + 4*θ.ωΛ0) / h(θ, y, x)] # dy_dx[1], dy_dx[2]
    return solve(ODEProblem(dy_dx, [log(ϕini), 0.0], (log(aini), 0.0))) # assumption: dlogϕ_dloga = 0 at a=aini
end

# relevant quantities as function of parameters and ODE state:
       logϕ(θ::Parameters, y, x) = y[1]
          ϕ(θ::Parameters, y, x) = exp(logϕ(θ, y, x))
dlogϕ_dloga(θ::Parameters, y, x) = fixed_point_iterate(y1′ -> y[2]*(1 + y1′ - θ.ω/6*(y1′)^2) / (θ.ωr0/a(x)^4 + θ.ωm0/a(x)^3 + θ.ωΛ0) / a(x)^3, 0.0) # solve y[2] = a^3 * (h*ϕ) * dlogϕ_dloga for y1′ = dlogϕ_dloga, with the expression for h expanded, using fixed-point iteration
          h(θ::Parameters, y, x) = √( 1/ϕ(θ, y, x) * (θ.ωr0/a(x)^4 + θ.ωm0/a(x)^3 + θ.ωΛ0) / (1 + dlogϕ_dloga(θ, y, x) - θ.ω/6*(dlogϕ_dloga(θ, y, x))^2) )

# find ϕini and ωΛ0 such that ϕ0=1
function shoot(ω=1e4, ωr0=5.5e-5*0.67^2, ωm0=0.317*0.67^2, ϕ0_target = 1.0; maxiters=100, verbose=false)
    @assert ϕ0_target > 0.0
    ϕini_lo = 0.0       # lower bound, guaranteed to have ϕ0 (= 0.0) < ϕ0_target
    ϕini_hi = ϕ0_target # upper bound, guaranteed to have ϕ0         > ϕ0_target

    ωΛ0 = 0.0 # extremely stupid initial *guess* (don't have any value of h0 to use)
    θ = Parameters(ω, ωr0, ωm0, ωΛ0)

    for iter in 1:maxiters
        # integrate scalar field with current cosmological parameters
        ϕini = (ϕini_lo + ϕini_hi) / 2
        y = integrate_ϕ(θ, ϕini)

        # present values of relevant quantities
        x0 = 0.0
        y0 = y(x0)
        ϕ0 = ϕ(θ, y0, x0)
        h0 = h(θ, y0, x0)
        ωϕ0 = -dlogϕ_dloga(θ, y0, x0) + θ.ω/6 * dlogϕ_dloga(θ, y0, x0)^2

        if verbose
            @printf "#%02d: JBD with ϕini = %.10f, ωΛ0 = %.10f has ϕ0 = %.10f, h0 = %.10f, ωϕ0 = %.10f\n" iter ϕini θ.ωΛ0 ϕ0 h0 ωϕ0
        end

        # we want the cosmology with (ϕini, ωΛ0) such that ϕ0 = 1
        if ϕini_lo ≈ ϕini_hi && ϕ0 ≈ 1.0
            return θ, ϕini # converged within maxiters
        end

        # update ωΛ0 to the value that satisfies the closure condition ∑(Ωi0) = 1 for the h0 in this cosmology
        # (then we must reiterate to correct the scalar field integral, which depends on ωΛ0)
        Ωr0, Ωm0, Ωϕ0 = ωr0/h0^2, ωm0/h0^2, ωϕ0/h0^2
        ΩΛ0 = 1 - Ωr0 - Ωm0 - Ωϕ0 # closure condition; equivalent to E = 1
        ωΛ0 = ΩΛ0*h0^2
        θ = Parameters(θ.ω, θ.ωr0, θ.ωm0, ωΛ0)

        # update ϕ0 (adjust aim), then shoot again
        if ϕ0 > ϕ0_target  # hit above,
            ϕini_hi = ϕini # so need lower  ϕini
        else               # hit below,
            ϕini_lo = ϕini # so need higher ϕini
        end
    end

    return nothing, NaN # did not converge within maxiters
end

shoot(verbose=true)

end
