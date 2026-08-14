import pytest

import dsgp4


OMM_FIELDS = {
    "CCSDS_OMM_VERS": "3.0",
    "OBJECT_NAME": "SENTINEL-1A",
    "OBJECT_ID": "2014-016A",
    "CENTER_NAME": "EARTH",
    "REF_FRAME": "TEME",
    "TIME_SYSTEM": "UTC",
    "MEAN_ELEMENT_THEORY": "SGP4",
    "EPOCH": "2022-02-28T01:57:54.918432",
    "MEAN_MOTION": "14.59199732",
    "ECCENTRICITY": "0.0001341",
    "INCLINATION": "98.1819",
    "RA_OF_ASC_NODE": "68.1874",
    "ARG_OF_PERICENTER": "82.4703",
    "MEAN_ANOMALY": "277.6657",
}


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("CENTER_NAME", "MARS", "CENTER_NAME must be EARTH"),
        ("REF_FRAME", "EME2000", "REF_FRAME must be TEME"),
        ("TIME_SYSTEM", "TAI", "TIME_SYSTEM must be UTC"),
    ],
)
def test_omm_rejects_unsupported_reference_context(field, value, expected):
    fields = dict(OMM_FIELDS, **{field: value})

    with pytest.raises(ValueError, match=expected):
        dsgp4.omm.OMM(fields)


def test_omm_reference_context_is_case_insensitive():
    fields = dict(OMM_FIELDS, CENTER_NAME="earth", REF_FRAME="teme", TIME_SYSTEM="utc")

    omm = dsgp4.omm.OMM(fields)

    assert omm.satellite_catalog_number == 0


def test_omm_missing_reference_context_uses_supported_defaults():
    fields = {
        key: value
        for key, value in OMM_FIELDS.items()
        if key not in {"CENTER_NAME", "REF_FRAME", "TIME_SYSTEM"}
    }

    omm = dsgp4.omm.OMM(fields)

    assert omm.satellite_catalog_number == 0
