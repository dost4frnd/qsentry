"""Generate the domain datasets (flattened CSVs) + integrity audits."""
from _common import (banner, data_dir, ensure_dirs, load_experiment,
                     resolve_domains, save_json)
from qsentry.datasets import (Preprocessor, audit, build_domain, save_domain,
                              write_audit)


def main():
    cfg, root = load_experiment()
    ensure_dirs(cfg)
    specs = resolve_domains(cfg, root)
    ddir = data_dir(cfg)
    banner(f"Generating {len(specs)} domains -> {ddir}")
    for name, spec in specs.items():
        df = build_domain(spec)
        path = ddir / f"tfqkd_{name}.csv"
        save_domain(df, path)
        pre = Preprocessor(seq_len=spec.seq_len)
        rep = audit(df, pre)
        write_audit(rep, ddir / f"audit_{name}.json")
        print(f"  {name:8s} rows={rep['rows']:5d} cols={rep['n_columns']} "
              f"dupes={rep['duplicate_rows']} leak={rep['split_leakage_ids']} "
              f"-> {path.name}")
    print("done.")


if __name__ == "__main__":
    main()
