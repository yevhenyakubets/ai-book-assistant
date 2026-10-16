def test_search_books(client):
    books = client.search_books("Foundation", limit=1)

    assert len(books) == 1
    assert books[0].title == "Foundation"
    assert books[0].key == "/works/OL46125W"


def test_fetch_work_description(client):
    desc = client.fetch_work_description("/works/OL46125W")

    assert desc == "An epic sci-fi masterwork."
