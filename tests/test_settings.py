from config.settings import get_settings


def test_settings_default_paths():
    settings = get_settings()
    assert settings.upload_path.name == "uploads"
    assert settings.vector_store_path.name == "vector_store"
