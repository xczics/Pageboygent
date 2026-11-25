from Pageboygent.storage import first_author_from_creators


def test_book_section_skips_editors_listed_first():
    creators = [
        {"creatorType": "editor", "lastName": "Harvey", "firstName": "Jason"},
        {"creatorType": "editor", "lastName": "Day", "firstName": "James"},
        {"creatorType": "author", "lastName": "Brenan", "firstName": "James M."},
        {"creatorType": "author", "lastName": "Bennett", "firstName": "Neil R."},
    ]
    assert first_author_from_creators(creators) == "Brenan"


def test_single_field_names_and_fallbacks():
    assert first_author_from_creators([{"creatorType": "author", "name": "IPCC"}]) == "IPCC"
    assert first_author_from_creators([{"creatorType": "author", "name": "Agnor, C"}]) == "Agnor"
    assert first_author_from_creators([{"creatorType": "author", "name": "Lunar and Planetary Institute, Houston"}]) \
        == "Lunar and Planetary Institute, Houston"
    assert first_author_from_creators([{"creatorType": "editor", "lastName": "Harvey"}]) == "Harvey"  # 只有编者
    assert first_author_from_creators([{"creatorType": "editor", "lastName": "Ed"},
                                       {"creatorType": "presenter", "lastName": "Talker"}]) == "Talker"
    assert first_author_from_creators([]) is None and first_author_from_creators(None) is None
