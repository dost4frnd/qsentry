import numpy as np
from qsentry.datasets import (DomainSpec, Preprocessor, audit, build_domain,
                              feature_columns, get_xy, split_frame)
from qsentry.physics import UNKNOWN_CLASS


def _small_clean():
    return build_domain(DomainSpec(name="clean", seq_len=32, n_train=80,
                                   n_val=32, n_test=80, seed=0))


def test_column_layout_388():
    df = _small_clean()
    assert len(feature_columns(32)) == 384
    assert df.shape[1] == 388  # 384 features + 4 metadata


def test_no_leakage_no_nans():
    df = _small_clean()
    pre = Preprocessor(seq_len=32)
    rep = audit(df, pre)
    assert rep["split_leakage_ids"] == 0
    assert rep["missing_values"] == 0
    assert rep["infinite_values"] == 0


def test_scaler_is_train_only_and_roundtrip(tmp_path):
    df = _small_clean()
    pre = Preprocessor(seq_len=32).fit(split_frame(df, "train"))
    p = tmp_path / "pre.npz"
    pre.save(p)
    pre2 = Preprocessor.load(p)
    assert np.allclose(pre.mean_, pre2.mean_)
    assert np.allclose(pre.std_, pre2.std_)
    x, y, labels = get_xy(df, "train", pre, known_only=True)
    assert x.shape[1:] == (32, 12)
    assert len(y) == len(labels) == x.shape[0]


def test_unknown_class_only_in_test():
    spec = DomainSpec(name="unknown", seq_len=32, n_train=80, n_val=32,
                      n_test=120, n_test_normal=12, mixed=True,
                      include_unknown_in_test=True, seed=1)
    df = build_domain(spec)
    train_labels = set(split_frame(df, "train")["label"])
    test_labels = set(split_frame(df, "test")["label"])
    assert UNKNOWN_CLASS not in train_labels
    assert UNKNOWN_CLASS in test_labels


def test_novelty_encoded_as_negative_one():
    pre = Preprocessor(seq_len=32)
    enc = pre.encode_labels(["normal", UNKNOWN_CLASS])
    assert enc[1] == -1
    assert enc[0] >= 0
