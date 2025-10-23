import subprocess


def run_command(command, description):
    """Run a shell command with printing and error handling."""
    try:
        print(f"Starting: {description}")
        subprocess.run(command, check=True)
        print(f"Completed: {description}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description} with error code {e.returncode}")
    except Exception as e:
        print(f"⚠️ Unexpected error while running {description}: {e}")


def main():

    # Run pipeline

    # Figures 6 and 7
    for benchmark in ["qasmbench-medium", "qasmbench-large", "queko-bss-16qbt", "queko-bss-54qbt", "queko-bss-81qbt"]:
        for backend in ["ibm_sherbrooke", "ankaa"]:
            run_command(["python", "run_benchmark.py", "--benchmark", benchmark,
                        "--backend", backend], f"Run Qlosure for {benchmark} on {backend}")

    # Figure 5
    run_command(["python", "run_benchmark.py", "--benchmark", "queko-bss-54qbt",
                "--backend", "imb_sherbrooke2X"], f"Run Qlosure for queko-bss-54qbt on imb_sherbrooke2X")

    # Tables 2 and 3
    run_command(["python", "run_baseline.py", "--benchmark", "queko-bss-16qbt",
                "--backend", "ibm_sherbrooke"], "Run Baselines for queko-bss-16qbt on ibm_sherbrooke")

    run_command(
        ["python", "generate-paper-plots/generate_figure_5.py"], "Generate Figure 5")

    for benchmark in ["queko-bss-16qbt", "queko-bss-54qbt", "queko-bss-81qbt"]:
        for backend in ["ibm_sherbrooke", "ankaa"]:
            run_command(["python", "generate-paper-plots/generate_figures_6_7.py", "--benchmark", benchmark,
                        "--backend", backend], f"Generate Figures 6 & 7 for {benchmark} on {backend}")

    run_command(["python", "generate-paper-plots/print_tables_2_3_4.py"],
                "Generate Tables 2, 3, 4")


if __name__ == "__main__":
    main()
