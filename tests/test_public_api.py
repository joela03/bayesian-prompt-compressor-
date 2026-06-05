from importlib.metadata import version


def test_public_api_exports():
    import prompt_compress

    assert prompt_compress.__version__ == version("prompt-compress")
    assert prompt_compress.PromptCompressor is not None
    assert prompt_compress.CompressionResult is not None
    assert prompt_compress.CompressionFailedError is not None
    assert prompt_compress.OptimisationConfig is not None
