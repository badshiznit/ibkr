import subprocess

def run_script(risk_reward_ratio, enable_prev_high_crossover, enable_prev_low_crossover, enable_prev_close_crossover, enable_premarket_high_crossover, enable_premarket_low_crossover):
    cmd = [
        "python", "trading_script.py",
        "--risk_reward_ratio", str(risk_reward_ratio)
    ]
    if enable_prev_high_crossover:
        cmd.append("--enable_prev_high_crossover")
    if enable_prev_low_crossover:
        cmd.append("--enable_prev_low_crossover")
    if enable_prev_close_crossover:
        cmd.append("--enable_prev_close_crossover")
    if enable_premarket_high_crossover:
        cmd.append("--enable_premarket_high_crossover")
    if enable_premarket_low_crossover:
        cmd.append("--enable_premarket_low_crossover")

    subprocess.run(cmd)

def main():
    risk_reward_ratios = [1, 2, 3]
    crossover_params = [
        {"enable_prev_high_crossover": True, "enable_prev_low_crossover": False, "enable_prev_close_crossover": False, "enable_premarket_high_crossover": False, "enable_premarket_low_crossover": False},
        {"enable_prev_high_crossover": False, "enable_prev_low_crossover": True, "enable_prev_close_crossover": False, "enable_premarket_high_crossover": False, "enable_premarket_low_crossover": False},
        {"enable_prev_high_crossover": False, "enable_prev_low_crossover": False, "enable_prev_close_crossover": True, "enable_premarket_high_crossover": False, "enable_premarket_low_crossover": False},
        {"enable_prev_high_crossover": False, "enable_prev_low_crossover": False, "enable_prev_close_crossover": False, "enable_premarket_high_crossover": True, "enable_premarket_low_crossover": False},
        {"enable_prev_high_crossover": False, "enable_prev_low_crossover": False, "enable_prev_close_crossover": False, "enable_premarket_high_crossover": False, "enable_premarket_low_crossover": True},
    ]

    for ratio in risk_reward_ratios:
        for params in crossover_params:
            run_script(
                risk_reward_ratio=ratio,
                enable_prev_high_crossover=params["enable_prev_high_crossover"],
                enable_prev_low_crossover=params["enable_prev_low_crossover"],
                enable_prev_close_crossover=params["enable_prev_close_crossover"],
                enable_premarket_high_crossover=params["enable_premarket_high_crossover"],
                enable_premarket_low_crossover=params["enable_premarket_low_crossover"]
            )

if __name__ == "__main__":
    main()
