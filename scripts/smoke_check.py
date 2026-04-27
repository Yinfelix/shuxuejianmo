from importlib import import_module
from importlib.metadata import version


PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("sympy", "sympy"),
    ("sklearn", "scikit-learn"),
    ("statsmodels", "statsmodels"),
    ("networkx", "networkx"),
    ("pulp", "PuLP"),
    ("openpyxl", "openpyxl"),
]


def main() -> int:
    failures = []

    for module_name, distribution_name in PACKAGES:
        try:
            import_module(module_name)
            print(f"{distribution_name}: {version(distribution_name)}")
        except Exception as exc:  # pragma: no cover - script style validation
            failures.append(f"{distribution_name}: {exc}")

    if failures:
        print("\nMissing or broken packages:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nCore workbench packages are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())