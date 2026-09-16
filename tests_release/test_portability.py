import json
from pathlib import Path

import pytest

import research_release as release


@pytest.fixture(autouse=True)
def local_settings(monkeypatch, tmp_path):
    monkeypatch.setenv('RESEARCH_PATHS_CONFIG', str(tmp_path / 'paths.yaml'))
    release.settings.cache_clear()
    yield
    release.settings.cache_clear()


def test_source_path_tracks_current_clone():
    assert Path(release.path('@repo/README.md')) == release.ROOT / 'README.md'
    assert Path(release.path('@data')).is_relative_to(release.ROOT)


def test_explicit_path_override(monkeypatch, tmp_path):
    file = tmp_path / 'paths.yaml'
    file.write_text(json.dumps({'path_overrides': {'@data/cohort': 'external/my-data'}}))
    assert Path(release.path('@data/cohort')) == release.ROOT / 'external/my-data'


def test_legacy_metadata_is_required_only_for_asset_resolution():
    source = 'identity: "@legacy:example:configs/base.yaml:identity"'
    with pytest.raises(ValueError, match='Historical compatibility metadata'):
        release.load_yaml(source)
    assert release.load_yaml(source, resolve_assets=False)['identity'].startswith('@legacy:')


def test_legacy_lookup_and_root_escape(monkeypatch, tmp_path):
    private = tmp_path / 'private'
    private.mkdir()
    (private / 'base.yaml').write_text('mode: synthetic-test\n')
    (tmp_path / 'paths.yaml').write_text(json.dumps({'legacy_config_roots': {'example': str(private)}}))
    assert release.path('@legacy:example:base.yaml:mode') == 'synthetic-test'
    with pytest.raises(ValueError, match='escaped'):
        release.path('@legacy:example:../outside.yaml:mode')


def test_remote_preparation_requires_explicit_host():
    with pytest.raises(ValueError, match='ssh.host'):
        release.ssh_host()


def test_local_configuration_can_defer_an_unused_remote(monkeypatch, tmp_path):
    assert release.load_yaml('remote: "@remote_registered"')['remote'] == '@remote_registered'
    (tmp_path / 'paths.yaml').write_text(json.dumps({
        'ssh': {'host': 'dataset-server'}, 'paths': {'remote_data': '/archive/organized'}
    }))
    release.settings.cache_clear()
    assert release.path('@remote_registered') == 'dataset-server:/archive/organized/03_registered_nifti'


def test_private_cohort_is_not_silently_used_for_real_configuration():
    with pytest.raises(ValueError, match='private cohort'):
        release.path('@private:mu_missing_baseline')
    assert release.private_value('mu_missing_baseline').startswith('PatientID_0000')
