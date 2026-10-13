from rl_x.runner.runner import Runner


if __name__ == "__main__":
    runner = Runner(implementation_package_names=["rl_x", "shape_your_body"])
    runner.run()
