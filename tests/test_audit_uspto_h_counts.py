from preprocess.audit_uspto_h_counts import audit_row


def test_audit_row_identifies_exactly_missing_oh_hydrogen():
    record = audit_row(
        {
            "smiles": "CCO",
            "h_nmr_peaks": [
                {"delta": 1.0, "nH": 3},
                {"delta": 3.5, "nH": 2},
            ],
            "c_nmr_peaks": [
                {"delta (ppm)": 20.0},
                {"delta (ppm)": 60.0},
            ],
        },
        row_index=0,
    )
    assert record["h_peak_count"] == 2
    assert record["h_shift_atom_count"] == 5
    assert record["oh_h_count"] == 1
    assert record["total_h_count"] == 6
    assert record["non_oh_h_count"] == 5
    assert record["total_minus_shift"] == 1
    assert record["non_oh_minus_shift"] == 0
    assert record["missing_equals_oh"]
    assert record["positive_missing_equals_oh"]
    assert record["accept_exact_or_missing_oh"]
    assert record["h_environment_count"] == 3
    assert record["h_peaks_minus_environments"] == -1
    assert record["total_c_count"] == 2
    assert record["c_environment_count"] == 2
    assert not record["c_line_count_gt_total_c"]


def test_audit_row_ignores_zero_integration_artifact_peak():
    record = audit_row(
        {
            "smiles": "CC",
            "h_nmr_peaks": [
                {"delta": 0.0, "nH": 0},
                {"delta": 1.0, "nH": 6},
            ],
            "c_nmr_peaks": [
                {"delta (ppm)": 10.0},
                {"delta (ppm)": 10.1},
            ],
        },
        row_index=0,
    )
    assert record["valid_integrations"]
    assert record["zero_integration_peak_count"] == 1
    assert record["h_shift_atom_count"] == 6
    assert record["matches_all_h"]
    assert not record["positive_missing_equals_oh"]
    assert record["accept_exact_or_missing_oh"]
    assert record["h_environment_count"] == 1
    assert record["h_peaks_minus_environments"] == 1
    assert record["c_line_count_gt_environments"]
