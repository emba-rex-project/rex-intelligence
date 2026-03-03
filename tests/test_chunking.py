from services.chunking import ChunkConfig, TextChunker


def test_chunker_enforces_overlap_less_than_size():
    config = ChunkConfig(chunk_size=1000, chunk_overlap=200)
    chunker = TextChunker(config)
    text = "This is a test." * 50
    chunks = chunker.chunk(text)
    assert chunks, "Expected at least one chunk"


def test_chunker_chunk_pages_preserves_order():
    config = ChunkConfig(chunk_size=10, chunk_overlap=2)
    chunker = TextChunker(config)
    pages = ["One two three four five", "Six seven eight nine ten"]
    chunk_records = chunker.chunk_pages(pages)
    page_numbers = {page for page, _, _ in chunk_records}
    assert page_numbers == {1, 2}
