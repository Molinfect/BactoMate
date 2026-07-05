from __future__ import annotations


def test_frozen_model_dirs_use_user_data_and_copy_legacy_files(monkeypatch, tmp_path):
    from bactomate.gui import paths

    app_root = tmp_path / "AppRoot"
    user_models = tmp_path / "UserData" / "Models"
    legacy_user = app_root / "omnipose_user_models"
    legacy_pretrained = app_root / "omnipose_pretrained"
    legacy_user.mkdir(parents=True)
    legacy_pretrained.mkdir(parents=True)
    (legacy_user / "custom_omni.pth").write_bytes(b"user model")
    (legacy_pretrained / "bact_phase_omnitorch_0").write_bytes(b"builtin")

    monkeypatch.setattr(paths, "_is_installed", lambda: False)
    monkeypatch.setattr(paths, "_is_frozen", lambda: True)
    monkeypatch.setattr(paths, "get_app_root", lambda: app_root)
    monkeypatch.setattr(paths, "_get_user_data_models_base", lambda: user_models)

    user_dir = paths.get_omnipose_user_models_dir()
    pretrained_dir = paths.get_omnipose_pretrained_cache_dir()

    assert user_dir == user_models / "omnipose_user_models"
    assert pretrained_dir == user_models / "omnipose_pretrained"
    assert (user_dir / "custom_omni.pth").read_bytes() == b"user model"
    assert (pretrained_dir / "bact_phase_omnitorch_0").read_bytes() == b"builtin"


def test_development_model_dirs_stay_repo_local(monkeypatch, tmp_path):
    from bactomate.gui import paths

    app_root = tmp_path / "repo"
    monkeypatch.setattr(paths, "_is_installed", lambda: False)
    monkeypatch.setattr(paths, "_is_frozen", lambda: False)
    monkeypatch.setattr(paths, "get_app_root", lambda: app_root)

    assert paths.get_omnipose_user_models_dir() == app_root / "omnipose_user_models"
    assert paths.get_omnipose_pretrained_cache_dir() == app_root / "omnipose_pretrained"


def test_windows_logs_fallback_to_appdata_when_project_logs_not_writable(monkeypatch, tmp_path):
    from bactomate.gui import paths

    app_root = tmp_path / "BactoMate"
    app_root.mkdir()
    user_logs = tmp_path / "AppData" / "BactoMate" / "Logs"

    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setattr(paths, "_is_installed", lambda: False)
    monkeypatch.setattr(paths, "_is_frozen", lambda: False)
    monkeypatch.setattr(paths, "get_app_root", lambda: app_root)
    monkeypatch.setattr(paths, "_get_user_data_logs_dir", lambda: user_logs)
    monkeypatch.setattr(paths, "_project_logs_dir_writable", lambda: None)

    assert paths.get_logs_dir() == user_logs


def test_windows_logs_use_project_folder_when_writable(monkeypatch, tmp_path):
    from bactomate.gui import paths

    app_root = tmp_path / "BactoMate"
    project_logs = app_root / "Logs"

    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setattr(paths, "_is_installed", lambda: False)
    monkeypatch.setattr(paths, "_is_frozen", lambda: False)
    monkeypatch.setattr(paths, "get_app_root", lambda: app_root)
    monkeypatch.setattr(paths, "_project_logs_dir_writable", lambda: project_logs)

    assert paths.get_logs_dir() == project_logs
