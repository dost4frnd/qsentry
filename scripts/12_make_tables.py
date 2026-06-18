"""Generate LaTeX tables from serialized results."""
from _common import (banner, load_experiment, load_json, metrics_dir,
                     resolve_domains, tab_dir)
from qsentry import tables


def main():
    cfg, root = load_experiment()
    td = str(tab_dir(cfg))
    md = metrics_dir(cfg)
    specs = resolve_domains(cfg, root)
    banner(f"Generating LaTeX tables -> {td}")
    made = []

    spec_dict = {}
    shift = {"clean": "none", "drift": "OU phase drift",
             "asym": "asymmetric loss", "unknown": "mixed + novelty"}
    for n, s in specs.items():
        spec_dict[n] = {"n_train": s.n_train, "n_val": s.n_val,
                        "n_test": s.n_test, "shift": shift.get(n, "--")}
    made.append(tables.write_dataset_table(spec_dict, td))

    try:
        cs = load_json(md / "closed_set.json")
        made.append(tables.write_closed_set_table(
            {k: v["metrics"] for k, v in cs.items()}, td,
            name="tab_closed_set_generated"))
    except FileNotFoundError:
        pass
    try:
        cd = load_json(md / "cross_domain.json")
        made.append(tables.write_cross_domain_table(cd, td,
                    name="tab_cross_domain_generated"))
    except FileNotFoundError:
        pass
    try:
        os_ = load_json(md / "open_set.json")
        made.append(tables.write_open_set_table(os_["report"], td,
                    name="tab_open_set_generated"))
    except FileNotFoundError:
        pass
    try:
        cal = load_json(md / "calibration.json")
        made.append(tables.write_calibration_table(cal["ece"], td))
    except FileNotFoundError:
        pass
    try:
        prof = load_json(md / "operating.json")
        made.append(tables.write_efficiency_table(prof, td))
    except FileNotFoundError:
        pass
    try:
        imp = load_json(md / "channel_importance.json")
        made.append(tables.write_channel_importance_table(imp, td))
    except FileNotFoundError:
        pass

    for p in made:
        print(f"  + {p}")
    print(f"{len(made)} tables written.")


if __name__ == "__main__":
    main()
