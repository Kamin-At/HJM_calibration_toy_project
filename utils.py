import numpy as np
from scipy import optimize
from scipy.stats import norm

EPSILON = 1e-7


def black_caplet_price(
    f: "(float) forward rate",
    k: "(float) strike rate",
    sigma: "(float) Black volatility",
    df: "(float) discount factor",
    t: "(float) time to reset date in years",
    tau: "(float) forward duration in years" = 0.25,
    N: "(float) notional amount" = 1.0,
):
    # return the caplet price
    # if k < 0.001 or t < 0.001 or sigma < 0.001 or f < 0.001:
    #     raise
    d1 = (np.log(f / k) + (sigma**2) * t * 0.5) / ((sigma + EPSILON) * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return df * N * tau * (f * norm.cdf(d1) - k * norm.cdf(d2))


def get_black_caplet_iv(
    price: "(float) caplet price",
    f: "(float) forward rate",
    k: "(float) strike rate",
    df: "(float) discount factor",
    t: "(float) time to reset date in years",
    tau: "(float) forward duration in years" = 0.25,
    N: "(float) notional amount" = 1.0,
    initial_guess: "(float) initial guess of the Black implied volatility" = 0.3,
):
    # return the implied volatility

    sigma = optimize.newton(
        lambda iv: price
        - black_caplet_price(f=f, k=k, sigma=iv, df=df, t=t, tau=tau, N=N),
        initial_guess,
    )

    return sigma


def black_cap_price(
    forward_curve: "(list[float]) forward curve",
    k: "(float) strike rate",
    sigma: "(float) Black volatility",
    zcb_prices: "(list[float]) zcb prices",
    time_to_reset_date: "(list[float]) times to reset date in years",
    taus: "(list[float]) forward durations in years",
    N: "(float) notional amount" = 1.0,
):
    # return the cap price

    assert (
        len(forward_curve) == len(zcb_prices)
        and len(zcb_prices) == len(time_to_reset_date)
        and len(time_to_reset_date) == len(taus)
    ), f"The legnths must be equal. len(forward_curve): {len(forward_curve)}, len(zcb_prices): {len(zcb_prices)}, len(time_to_reset_date): {len(zcb_prices)}, len(taus): {len(taus)}"

    cap_price = 0

    for i in range(len(zcb_prices)):
        caplet_price = black_caplet_price(
            forward_curve[i],
            k,
            sigma,
            zcb_prices[i],  # Discount at maturity not reset date
            time_to_reset_date[i],
            taus[i],
            N,
        )
        cap_price += caplet_price

    return cap_price


def get_black_cap_iv(
    price: "(float) caplet price",
    forward_curve: "(float) forward rate",
    k: "(float) strike rate",
    zcb_prices: "(float) discount factor",
    time_to_reset_date: "(float) time to reset date in years",
    taus: "(float) forward duration in years" = 0.25,
    N: "(float) notional amount" = 1.0,
    initial_guess: "(float) initial guess of the Black implied volatility" = 0.3,
):
    # return the implied volatility

    sigma = optimize.newton(
        lambda iv: price
        - black_cap_price(
            forward_curve, k, iv, zcb_prices, time_to_reset_date, taus, N
        ),
        initial_guess,
    )

    return sigma


def spot_curve_to_zcb_curve(
    spot_curve: "(list[float]) spot curve", tenors: "(list[float]) tenors"
):
    # return zcb curve => list[float]

    assert len(spot_curve) == len(
        tenors
    ), f"len(spot_curve) [{len(spot_curve)}] != len(tenors) [{len(tenors)}]"

    zcb_curve = []
    for i in range(len(spot_curve)):
        zcb_curve.append(1.0 / (1.0 + spot_curve[i] * tenors[i]))

    return zcb_curve


def zcb_curve_to_spot_curve(
    zcb_curve: "(list[float]) zcb curve", tenors: "(list[float]) tenors"
):
    # return spot curve => list[float]

    assert len(zcb_curve) == len(
        tenors
    ), f"len(zcb_curve) [{len(zcb_curve)}] != len(tenors) [{len(tenors)}]"

    spot_curve = []
    for i in range(len(zcb_curve)):
        spot_curve.append((1.0 / zcb_curve[i] - 1.0) / tenors[i])

    return spot_curve


def zcb_curve_to_forward_curve(
    zcb_curve: "(list[float]) zcb curve", tenors: "(list[float]) tenors"
):
    # return forward curve => list[float]

    assert len(zcb_curve) == len(
        tenors
    ), f"len(zcb_curve) [{len(zcb_curve)}] != len(tenors) [{len(tenors)}]"

    zcb_curve = [1.0] + zcb_curve
    tenors = [0.0] + tenors

    forward_curve = []
    for i in range(len(zcb_curve) - 1):
        dt = tenors[i + 1] - tenors[i]
        forward_curve.append((zcb_curve[i] / zcb_curve[i + 1] - 1.0) / dt)
    forward_curve.append(np.nan)
    return forward_curve


def zcb_curve_to_forward_swap_curve(
    zcb_curve: "(list[float]) zcb curve", tenors: "(list[float]) tenors"
):
    # return forward swap curve => list[float]

    assert len(zcb_curve) == len(
        tenors
    ), f"len(zcb_curve) [{len(zcb_curve)}] != len(tenors) [{len(tenors)}]"

    zcb_curve = [1.0] + zcb_curve
    tenors = [0.0] + tenors
    dt = [(tenors[i + 1] - tenors[i]) for i in range(len(tenors) - 1)]
    zcb_x_dt_cumsum_curve = np.cumsum(
        [dt[i] * zcb_curve[i + 1] for i in range(1, len(zcb_curve) - 1)]
    )
    delta_zcb = [zcb_curve[1] - zcb_curve[i] for i in range(2, len(zcb_curve))]

    assert len(zcb_x_dt_cumsum_curve) == len(
        delta_zcb
    ), f"len(zcb_x_dt_cumsum_curve) [{len(zcb_x_dt_cumsum_curve)}] != len(delta_zcb) [{len(delta_zcb)}]"

    forward_swap_curve = (
        [np.nan]
        + [delta_zcb[i] / zcb_x_dt_cumsum_curve[i] for i in range(len(delta_zcb))]
        + [np.nan]
    )

    return forward_swap_curve
