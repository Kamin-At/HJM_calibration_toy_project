import numpy as np
from utils import *
import pandas as pd
from scipy.optimize import minimize


class HJM_Model:
    def __init__(
        self,
        family: '(str) choose from ["HW", "HL", "MM"]',
    ):
        self.family = family

    def __cal_vol_HL(
        self, T: "(float or np.array) tenor", params: "(np.array) model parameters"
    ):
        assert len(params) == 1, "HL requires 1 parameters"

        return params[0] * T / T

    def __cal_vol_HW(
        self, T: "(float or np.array) tenor", params: "(np.array) model parameters"
    ):
        assert len(params) == 2, "HW requires 2 parameters"

        return params[0] * np.exp(-params[1] * T)

    def __cal_vol_MM(
        self, T: "(float or np.array) tenor", params: "(np.array) model parameters"
    ):
        assert len(params) == 3, "HW requires 3 parameters"

        return (params[0] + params[1] * T) * np.exp(-params[2] * T)

    def cal_vol(
        self, T: "(float or np.array) tenor", params: "(np.array) model parameters"
    ):

        if self.family == "HL":
            return self.__cal_vol_HL(T, params)
        elif self.family == "HW":
            return self.__cal_vol_HW(T, params)
        elif self.family == "MM":
            return self.__cal_vol_MM(T, params)
        else:
            raise NotImplementedError(
                'Model not found => choose from ["HW", "HL", "MM"]'
            )

    def __cal_integral_G(
        self, T: "(float or np.array) tenor", params: "(np.array) model parameters"
    ):
        # Calculate the integral [from 0 to T] of G(s) ds
        if self.family == "HL":
            return params[0] * T + 0.5 * params[1] * T**2
        elif self.family == "HW":
            return (
                params[0] * (1 - np.exp(-params[2] * T))
                + 0.5 * params[1] * (1 - np.exp(-2 * params[2] * T))
            ) / params[2]
        elif self.family == "MM":
            return (
                T**2
                * np.exp(-params[5] - 2)
                * (
                    6 * params[0] * np.exp(2)
                    + 4 * params[1] * np.exp(2) * T
                    + 6 * params[2] * params[5] * np.exp(params[5])
                    + params[5]
                    * T
                    * np.exp(params[5])
                    * (4 * params[3] + 3 * params[4] * T)
                )
                / 12
            )
        else:
            raise NotImplementedError(
                'family G not found => choose from ["HW", "HL", "MM"]'
            )

    def cal_zcb_from_G(
        self, T: "(float or np.array) tenor", g_params: "(np.array) model parameters"
    ):
        if type(T) == list:
            return np.exp(-self.__cal_integral_G(np.array(T), g_params))
        else:
            return np.exp(-self.__cal_integral_G(T, g_params))

    def evaluate_zcb_fit_loss(
        self,
        g_params: "(np.array) parameters",
        zcb_curve: "(list[float]) zcb curve",
        tenors: "(list[float]) tenors in years",
    ):
        # Compare model ZCB curve with the market ZCB
        total_loss = np.mean(
            (np.log(zcb_curve) - np.log(self.cal_zcb_from_G(tenors, g_params))) ** 2
        )

        return total_loss

    def evaluate_cap_pricing_fit_loss(
        self,
        params: "(np.array) parameters",
        g_params: "(np.array) parameters",
        cap_vols: "(list[float]) cap vols from the market",
        forward_swap_curve: "(list[float]) forward swap curve",
        time_to_reset_date: "(list[float]) time to reset dates",
        taus: "(list[float]) time from reset date to maturity date => default value: 0.25",
        tenors: "(list[float]) tenors in years",
    ):
        assert len(cap_vols) == len(
            time_to_reset_date
        ), f"len(cap_vols) [{len(cap_vols)}] != len(time_to_reset_date) [{len(time_to_reset_date)}]"

        assert len(cap_vols) == len(
            forward_swap_curve
        ), f"len(cap_vols) [{len(cap_vols)}] != len(forward_swap_curve) [{len(forward_swap_curve)}]"

        model_zcb_curve = self.cal_zcb_from_G(tenors, g_params)

        model_forward_curve = zcb_curve_to_forward_curve(
            model_zcb_curve[1:], tenors[1:]
        )
        if sum(np.array(model_forward_curve) <= 0.001):
            return 100000  # Constraint for extreme case

        total_loss = 0.0

        model_sigmas = self.cal_vol(time_to_reset_date, params)

        if sum(self.cal_vol(time_to_reset_date, params) <= 0):
            return 100000  # Constraint for extreme case

        for i in range(len(cap_vols)):
            model_cap_price = black_cap_price(
                forward_curve=model_forward_curve[:-1],
                k=forward_swap_curve[i],
                sigma=model_sigmas[i],
                zcb_prices=model_zcb_curve[2:],
                time_to_reset_date=time_to_reset_date,
                taus=taus[1:-1],
                N=1.0,
            )

            market_cap_price = black_cap_price(
                forward_curve=model_forward_curve[:-1],
                k=forward_swap_curve[i],
                sigma=cap_vols[i],
                zcb_prices=model_zcb_curve[2:],
                time_to_reset_date=time_to_reset_date,
                taus=taus[1:-1],
                N=1.0,
            )
            total_loss += (np.log(model_cap_price) - np.log(market_cap_price)) ** 2

        return total_loss / len(cap_vols)

    def evaluate_cap_prices(
        self,
        params: "(np.array) parameters",
        g_params: "(np.array) parameters",
        cap_vols: "(list[float]) cap vols from the market",
        forward_swap_curve: "(list[float]) forward swap curve",
        time_to_reset_date: "(list[float]) time to reset dates",
        taus: "(list[float]) time from reset date to maturity date => default value: 0.25",
        tenors: "(list[float]) tenors in years",
    ):
        assert len(cap_vols) == len(
            time_to_reset_date
        ), f"len(cap_vols) [{len(cap_vols)}] != len(time_to_reset_date) [{len(time_to_reset_date)}]"

        assert len(cap_vols) == len(
            forward_swap_curve
        ), f"len(cap_vols) [{len(cap_vols)}] != len(forward_swap_curve) [{len(forward_swap_curve)}]"

        model_zcb_curve = self.cal_zcb_from_G(tenors, g_params)

        model_forward_curve = zcb_curve_to_forward_curve(
            model_zcb_curve[1:], tenors[1:]
        )

        model_sigmas = self.cal_vol(time_to_reset_date, params)

        model_cap_prices = []
        model_iv = []
        market_cap_prices = []
        market_iv = []

        for i in range(len(cap_vols)):
            model_cap_price = black_cap_price(
                forward_curve=model_forward_curve[:-1],
                k=forward_swap_curve[i],
                sigma=model_sigmas[i],
                zcb_prices=model_zcb_curve[2:],
                time_to_reset_date=time_to_reset_date,
                taus=taus[1:-1],
                N=1.0,
            )

            model_cap_prices.append(model_cap_price)
            model_iv.append(
                get_black_cap_iv(
                    price=model_cap_price,
                    forward_curve=model_forward_curve[:-1],
                    k=forward_swap_curve[i],
                    zcb_prices=model_zcb_curve[2:],
                    time_to_reset_date=time_to_reset_date,
                    taus=taus[1:-1],
                    N=1.0,
                    initial_guess=0.3,
                )
            )

            market_cap_price = black_cap_price(
                forward_curve=model_forward_curve[:-1],
                k=forward_swap_curve[i],
                sigma=cap_vols[i],
                zcb_prices=model_zcb_curve[2:],
                time_to_reset_date=time_to_reset_date,
                taus=taus[1:-1],
                N=1.0,
            )

            market_cap_prices.append(market_cap_price)
            market_iv.append(
                get_black_cap_iv(
                    price=market_cap_price,
                    forward_curve=model_forward_curve[:-1],
                    k=forward_swap_curve[i],
                    zcb_prices=model_zcb_curve[2:],
                    time_to_reset_date=time_to_reset_date,
                    taus=taus[1:-1],
                    N=1.0,
                    initial_guess=0.3,
                )
            )
        return model_iv, market_iv

    def calibrate_zcb_curve(
        self,
        zcb_curve: "(list[float]) zcb curve",
        tenors: "(list[float]) tenors in years",
        initial_params: "(np.array) initial parameters",
    ):
        # Find the parameters zeta that makes G(zeta) consistent with the zcb curve
        assert len(zcb_curve) == len(
            tenors
        ), f"len(zcb_curve) [{len(zcb_curve)}] != len(tenors) [{len(tenors)}]"

        result = minimize(
            lambda x: self.evaluate_zcb_fit_loss(
                x,
                zcb_curve,
                tenors,
            ),
            initial_params,
            method="Nelder-Mead",
            options={"maxiter": 2000},
            tol=1e-10,
        )
        assert result.success, f"Optimization issue: {result}"
        return result

    def calibrate_cap_price(
        self,
        cap_vols: "(list[float]) cap vols",
        forward_swap_curve: "(list[float]) forward swap curve",
        time_to_reset_date: "(list[float]) time to reset dates",
        taus: "(list[float]) time from reset date to maturity date => default value: 0.25",
        tenors: "(list[float]) tenors in years",
        initial_params: "(np.array) initial parameters",
        g_params: "(np.array) G parameters",
    ):

        result = minimize(
            lambda x: self.evaluate_cap_pricing_fit_loss(
                x,
                g_params,
                cap_vols,
                forward_swap_curve,
                time_to_reset_date,
                taus,
                tenors,
            ),
            initial_params,
            method="Nelder-Mead",
            options={"maxiter": 2000},
            tol=1e-8,
        )
        assert result.success, f"Optimization issue: {result}"
        return result


if __name__ == "__main__":
    pass
