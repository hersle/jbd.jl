module JBD

using DifferentialEquations
using Printf

struct Parameters
    # TODO: what if I give physical parameters instead???
    # TODO: how is h a derived parameter?
    ω::Float64   # Brans-Dicke coupling
    Ωr0::Float64 # "unphysical" radiation   density parameter
    Ωm0::Float64 # "unphysical" matter      density parameter
    ΩΛ0::Float64 # "unphysical" dark energy density parameter
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
#   y[1] = logϕ
#   y[2] = a^3 * E * ϕ * dlogϕ/dloga
#   dy_dx[1] = dlogϕ / dloga
#   dy_dx[2] = d(a^3*E*ϕ*dlogϕ/dloga) / dloga = 3 * a^3 * (Ωm0/a^3 + 4*ΩΛ0) / ((3 + 2ω) * E)
function integrate_ϕ(θ::Parameters, ϕini; aini=1e-10)
    dy_dx(y, _, x) = [dlogϕ_dloga(θ, y, x), 3 / (3+2*θ.ω) * a(x)^3 * (θ.Ωm0/a(x)^3 + 4*θ.ΩΛ0) / E(θ, y, x)] # dy_dx[1], dy_dx[2]
    return solve(ODEProblem(dy_dx, [log(ϕini), 0.0], (log(aini), 0.0)))
end

# relevant quantities as function of parameters and ODE state:
       logϕ(θ::Parameters, y, x) = y[1]
          ϕ(θ::Parameters, y, x) = exp(logϕ(θ, y, x))
dlogϕ_dloga(θ::Parameters, y, x) = fixed_point_iterate(y1′ -> y[2]*(1 + y1′ - θ.ω/6*(y1′)^2) / (θ.Ωr0/a(x)^4 + θ.Ωm0/a(x)^3 + θ.ΩΛ0) / a(x)^3, 0.0) # solve y[2] = a^3 * (E*ϕ) * dlogϕ_dloga for y1′ = dlogϕ_dloga, with the expression for E expanded, using fixed-point iteration
          E(θ::Parameters, y, x) = √( 1/ϕ(θ, y, x) * (θ.Ωr0/a(x)^4 + θ.Ωm0/a(x)^3 + θ.ΩΛ0) / (1 + dlogϕ_dloga(θ, y, x) - θ.ω/6*(dlogϕ_dloga(θ, y, x))^2) )

# find ϕini and ΩΛ0 such that ϕ0=1 and E0=1
function shoot(ω=1e2, Ωr0=5.5e-5, Ωm0=0.317, ϕ0_target = 1.0; maxiters=100, verbose=false)
    @assert ϕ0_target > 0.0
    ϕini_lo = 0.0       # lower bound, guaranteed to have ϕ0 (= 0.0) < ϕ0_target
    ϕini_hi = ϕ0_target # upper bound, guaranteed to have ϕ0         > ϕ0_target

    ΩΛ0 = 1 - Ωr0 - Ωm0 # initial *guess* for ΩΛ0 (based on sum(Ωi0)=1 and Ωϕ0≈0)
    θ = Parameters(ω, Ωr0, Ωm0, ΩΛ0)

    for iter in 1:maxiters
        # integrate scalar field with current cosmological parameters
        ϕini = (ϕini_lo + ϕini_hi) / 2
        y = integrate_ϕ(θ, ϕini)

        # present values of relevant quantities
        x0 = 0.0
        y0 = y(x0)
        ϕ0 = ϕ(θ, y0, x0)
        E0 = E(θ, y0, x0)
        Ωϕ0 = -dlogϕ_dloga(θ, y0, x0) + θ.ω/6 * dlogϕ_dloga(θ, y0, x0)^2
        Ω0 = Ωr0 + Ωm0 + ΩΛ0 + Ωϕ0

        if verbose
            @printf "#%02d: JBD with ϕini = %.10f, ΩΛ0 = %.10f has ϕ0 = %.10f, E0 = %.10f, Ωϕ0 = %.10f, ∑(Ωi0) = %.10f\n" iter ϕini θ.ΩΛ0 ϕ0 E0 Ωϕ0 Ω0
        end

        # we want the cosmology with (ϕini, ΩΛ0) such that ϕ0 = 1 and E0 = 1
        if ϕini_lo ≈ ϕini_hi && E0 ≈ 1.0 && ϕ0 ≈ 1.0
            return θ, ϕini # converged within maxiters
        end

        # update ΩΛ0 (so next cosmology has E closer to 1)
        ΩΛ0 = 1 - θ.Ωm0 - θ.Ωr0 - Ωϕ0 # equivalent to E = 1
        θ = Parameters(θ.ω, θ.Ωr0, θ.Ωm0, ΩΛ0)

        # update ϕ0 (adjust aim) and shoot again
        if ϕ0 > ϕ0_target
            ϕini_hi = ϕini # hit above, so need lower  ϕini
        else
            ϕini_lo = ϕini # hit below, so need higher ϕini
        end
    end

    return nothing, NaN # did not converge within maxiters
end

shoot(verbose=true)

end
