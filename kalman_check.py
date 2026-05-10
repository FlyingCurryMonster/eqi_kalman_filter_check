"""
Data-driven check of EQI eq. (2.29).

Two filter variants share the update step (2.26)/(2.27). They differ only in
how the predicted mean x̂_{t+1|t} is formed:

  (2.29) as printed with t|t:    x̂_{t+1|t} = A x̂_{t|t}   + A Σ̂_{t|t}   Bᵀ (B Σ̂_{t|t}   Bᵀ + Σ_η)⁻¹ (y_t − B x̂_{t|t})
  (2.29) with t|t-1:             x̂_{t+1|t} = A x̂_{t|t-1} + A Σ̂_{t|t-1} Bᵀ (B Σ̂_{t|t-1} Bᵀ + Σ_η)⁻¹ (y_t − B x̂_{t|t-1})

Both forms claim to follow from substituting (2.27) into (2.24); only the
second is internally consistent (the first double-applies the innovation).
"""
import numpy as np
import matplotlib.pyplot as plt


def simulate(A, b_const, B, tau_eps, tau_eta, T, x0, seed):
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    y = np.empty(T)
    xt = x0
    for t in range(T):
        xt = A * xt + b_const + tau_eps * rng.standard_normal()
        x[t] = xt
        y[t] = B * xt + tau_eta * rng.standard_normal()
    return x, y


def run_filter(y, A, b_const, B, tau_eps, tau_eta, sig_prior, x_prior, variant):
    T = len(y)
    Se = tau_eps ** 2
    Sn = tau_eta ** 2

    x_pred = x_prior
    S_pred = sig_prior

    x_post_arr = np.empty(T)
    S_post_arr = np.empty(T)
    x_pred_arr = np.empty(T)
    S_pred_arr = np.empty(T)
    loglik = 0.0

    for t in range(T):
        # Predictive density of y_t: N(B*x_pred, B*S_pred*B + Sn)
        innov = y[t] - B * x_pred
        S_innov = B * S_pred * B + Sn
        loglik += -0.5 * (np.log(2 * np.pi * S_innov) + innov ** 2 / S_innov)
        x_pred_arr[t] = x_pred
        S_pred_arr[t] = S_pred

        # Update step (2.26)/(2.27) — same for both variants
        K = S_pred * B / S_innov
        x_post = x_pred + K * innov
        S_post = (1 - K * B) * S_pred
        x_post_arr[t] = x_post
        S_post_arr[t] = S_post

        # Prediction step
        S_pred = A * S_post * A + Se  # (2.28), same in both
        if variant == "ttm1":
            # (2.29) with t|t-1: A x̂_{t|t-1} + A Σ̂_{t|t-1} Bᵀ(...)⁻¹(y_t − B x̂_{t|t-1})
            # = A x̂_{t|t} after substituting (2.27); this is the simpler equivalent.
            x_pred = A * x_post + b_const
        elif variant == "tt":
            # (2.29) as printed with t|t — uses posterior cov & posterior mean in residual
            S_innov_post = B * S_post * B + Sn
            x_pred = (
                A * x_post
                + A * S_post * B / S_innov_post * (y[t] - B * x_post)
                + b_const
            )
        else:
            raise ValueError(variant)

    return {
        "x_post": x_post_arr,
        "S_post": S_post_arr,
        "x_pred": x_pred_arr,
        "S_pred": S_pred_arr,
        "loglik": loglik,
    }


def riccati_muth(tau_eps, tau_eta):
    # σ̂² = ½ τ_ε² (1 + sqrt((2κ)² + 1)),  κ = τ_η/τ_ε
    kappa = tau_eta / tau_eps
    return 0.5 * tau_eps ** 2 * (1 + np.sqrt((2 * kappa) ** 2 + 1))


def riccati_ar1(a, tau_eps, tau_eta):
    # (1-a²) σ² + a² σ⁴ / (σ² + τ_η²) = τ_ε²  → quadratic in σ²
    Se = tau_eps ** 2
    Sn = tau_eta ** 2
    # (1-a²)σ⁴ + [(1-a²)Sn - Se + a²·... ] simplest: solve numerically
    # Equivalent closed form from book: σ² = ½[(a²-1)Sn + Se + sqrt(((a²-1)Sn+Se)² + 4 Sn Se)]
    disc = ((a ** 2 - 1) * Sn + Se) ** 2 + 4 * Sn * Se
    return 0.5 * ((a ** 2 - 1) * Sn + Se + np.sqrt(disc))


def run_example(name, A, b_const, B, tau_eps, tau_eta, sig_init, T=5000, seed=0,
                analytic_sig=None):
    x_true, y = simulate(A, b_const, B, tau_eps, tau_eta, T, x0=0.0, seed=seed)

    # Initialize at stationary prior if provided, else 1.0
    sig_prior = sig_init if sig_init is not None else 1.0

    res_c = run_filter(y, A, b_const, B, tau_eps, tau_eta, sig_prior, 0.0, "ttm1")
    res_a = run_filter(y, A, b_const, B, tau_eps, tau_eta, sig_prior, 0.0, "tt")

    rmse_c = np.sqrt(np.mean((res_c["x_post"] - x_true) ** 2))
    rmse_a = np.sqrt(np.mean((res_a["x_post"] - x_true) ** 2))

    K_c_ss = res_c["S_pred"][-1] * B / (B * res_c["S_pred"][-1] * B + tau_eta ** 2)
    K_a_ss = res_a["S_pred"][-1] * B / (B * res_a["S_pred"][-1] * B + tau_eta ** 2)

    print(f"\n=== {name} ===")
    print(f"  T = {T},  τ_ε = {tau_eps},  τ_η = {tau_eta},  A = {A},  b = {b_const}")
    if analytic_sig is not None:
        print(f"  analytic steady-state Σ_{{t+1|t}} (Riccati) = {analytic_sig:.6f}")
        print(f"    (2.29) with t|t-1            Σ_pred[T-1] = {res_c['S_pred'][-1]:.6f}")
        print(f"    (2.29) as printed with t|t   Σ_pred[T-1] = {res_a['S_pred'][-1]:.6f}")
    print(f"  state RMSE   (2.29) t|t-1 = {rmse_c:.4f}    (2.29) as printed t|t = {rmse_a:.4f}")
    print(f"  loglik       (2.29) t|t-1 = {res_c['loglik']:.2f}   (2.29) as printed t|t = {res_a['loglik']:.2f}")
    print(f"  Δ loglik ( t|t-1  −  as printed t|t ) = {res_c['loglik'] - res_a['loglik']:.2f}")
    print(f"  steady-state K  (2.29) t|t-1 = {K_c_ss:.4f}   (2.29) as printed t|t = {K_a_ss:.4f}")

    # Per-step paired statistics — significance of the gap.
    tau_eta2 = tau_eta ** 2
    Sc_pred = res_c["S_pred"] + tau_eta2
    Sa_pred = res_a["S_pred"] + tau_eta2
    ll_c = -0.5 * (np.log(2 * np.pi * Sc_pred) + (y - res_c["x_pred"]) ** 2 / Sc_pred)
    ll_a = -0.5 * (np.log(2 * np.pi * Sa_pred) + (y - res_a["x_pred"]) ** 2 / Sa_pred)
    d_ll = ll_c - ll_a
    # squared-error difference (corrected has lower SE if positive)
    d_se = (res_a["x_post"] - x_true) ** 2 - (res_c["x_post"] - x_true) ** 2

    def hac_se(d, L=None):
        """Newey-West HAC standard error of sum_t d_t. L = bandwidth."""
        T = len(d)
        if L is None:
            L = int(np.floor(4 * (T / 100) ** (2 / 9)))
        m = d.mean()
        u = d - m
        gamma0 = (u * u).mean()
        s = gamma0
        for k in range(1, L + 1):
            w = 1 - k / (L + 1)
            gk = (u[k:] * u[:-k]).mean()
            s += 2 * w * gk
        # variance of sample mean ≈ s/T, so SE of sum = sqrt(T*s)
        return np.sqrt(T * s), L

    se_ll, L_ll = hac_se(d_ll)
    se_se, L_se = hac_se(d_se)
    sum_ll = d_ll.sum()
    sum_se = d_se.sum()
    print(f"  paired Δ loglik (t|t-1 − as printed t|t) = {sum_ll:+.2f}  HAC SE (L={L_ll}) = {se_ll:.2f}  →  {sum_ll/se_ll:+.1f}σ")
    print(f"  paired Σ(SE_asprinted − SE_t|t-1)        = {sum_se:+.2f}  HAC SE (L={L_se}) = {se_se:.2f}  →  {sum_se/se_se:+.1f}σ")

    return x_true, y, res_c, res_a


def plot_example(name, x_true, y, res_c, res_a, fname):
    win = slice(0, 200)
    fig, axes = plt.subplots(4, 1, figsize=(10, 11))

    axes[0].plot(x_true[win], "k-", lw=1.2, label="true x_t")
    axes[0].plot(res_c["x_post"][win], "b-", lw=1.0, label="(2.29) with t|t-1")
    axes[0].plot(res_a["x_post"][win], "r--", lw=1.0, label="(2.29) as printed with t|t")
    axes[0].plot(y[win], "k.", ms=2, alpha=0.3, label="y_t")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].set_title(f"{name}: filtered state (first 200 steps)")

    axes[1].plot(np.sqrt(res_c["S_post"]), "b-", label="(2.29) with t|t-1")
    axes[1].plot(np.sqrt(res_a["S_post"]), "r--", label="(2.29) as printed with t|t")
    axes[1].set_ylabel("posterior std dev  √Σ̂_{t|t}")
    axes[1].legend(fontsize=8)

    innov_c = y - res_c["x_pred"]
    innov_a = y - res_a["x_pred"]
    axes[2].plot(innov_c[win], "b-", lw=0.8, label="(2.29) with t|t-1")
    axes[2].plot(innov_a[win], "r--", lw=0.8, label="(2.29) as printed with t|t")
    axes[2].axhline(0, color="k", lw=0.5)
    # axes[2].set_ylabel("innovation  y_t − B x̂_{t|t-1}")
    axes[2].set_ylabel(r"innovation  $y_{t} - \hat{x}_{t+1|t}$")
    axes[2].legend(fontsize=8)

    tau_eta2 = res_c["_tau_eta2"]
    Sc = res_c["S_pred"] + tau_eta2
    Sa = res_a["S_pred"] + tau_eta2
    ll_c = -0.5 * (np.log(2 * np.pi * Sc) + innov_c ** 2 / Sc)
    ll_a = -0.5 * (np.log(2 * np.pi * Sa) + innov_a ** 2 / Sa)
    axes[3].plot(np.cumsum(ll_c - ll_a), "g-")
    axes[3].axhline(0, color="k", lw=0.5)
    axes[3].set_ylabel("cum loglik:  t|t-1  −  as printed t|t")
    axes[3].set_xlabel("t")

    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  → saved {fname}")


def main():
    # ---- Example 2.1: Muth / random walk + noise ----
    # κ = τ_η/τ_ε = 0.5  →  K ≈ 0.83  (intermediate gain to make the (2.29) typo bite hardest)
    tau_eps, tau_eta = 1.0, 0.5
    sig_ss_muth = riccati_muth(tau_eps, tau_eta)
    x_true, y, res_c, res_a = run_example(
        "Example 2.1 (Muth: random walk + noise)",
        A=1.0, b_const=0.0, B=1.0,
        tau_eps=tau_eps, tau_eta=tau_eta,
        sig_init=sig_ss_muth, T=5000, seed=1,
        analytic_sig=sig_ss_muth,
    )
    res_c["_tau_eta2"] = tau_eta ** 2
    res_a["_tau_eta2"] = tau_eta ** 2
    plot_example("Example 2.1 (Muth)", x_true, y, res_c, res_a,
                 "/home/rakin/rnb76-rclone/datasci-mini-projects/eqi_kalman_filter_check/example_2_1.png")

    # ---- Example 2.2: AR(1) + noise ----
    lam = 0.1
    mu = 0.0
    a = 1 - lam
    b_const = lam * mu
    # κ = 0.5 here too; previously κ=20 made K≈0.012 and the two filters indistinguishable
    tau_eps, tau_eta = 1.0, 0.5
    sig_ss_ar1 = riccati_ar1(a, tau_eps, tau_eta)
    x_true, y, res_c, res_a = run_example(
        "Example 2.2 (AR(1) + noise)",
        A=a, b_const=b_const, B=1.0,
        tau_eps=tau_eps, tau_eta=tau_eta,
        sig_init=sig_ss_ar1, T=5000, seed=2,
        analytic_sig=sig_ss_ar1,
    )
    res_c["_tau_eta2"] = tau_eta ** 2
    res_a["_tau_eta2"] = tau_eta ** 2
    plot_example("Example 2.2 (AR(1))", x_true, y, res_c, res_a,
                 "/home/rakin/rnb76-rclone/datasci-mini-projects/eqi_kalman_filter_check/example_2_2.png")


if __name__ == "__main__":
    main()
