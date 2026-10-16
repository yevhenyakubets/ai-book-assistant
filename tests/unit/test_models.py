from ai_book_assistant.models import Book


def test_book_from_search_doc(sample_search_doc):
    book = Book.from_search_doc(sample_search_doc)

    assert book.title == "Foundation"
    assert book.authors == ["Isaac Asimov"]
    assert book.first_publish_year == 1951
    assert book.key == "/works/OL46125W"
    assert book.subjects == ["Science Fiction"]
    assert book.covers == [14612610]
