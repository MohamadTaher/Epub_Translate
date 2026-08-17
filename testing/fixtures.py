"""
The books the suite uploads, built here rather than committed.

A test book has to be small enough to translate for a few cents and specific
enough to prove something: `zh_small` carries a cover page with no text (which
must not be counted as a chapter) and recurring names (which must reach the
glossary), `zh_mixed` is half already in English (which must be detected and
skipped), `zh_plain` has neither an NCX nor a nav document (which the writer has
to add exactly once), and `padded` is only large so the upload limit has
something to refuse.

Generated, not checked in: `.gitignore` already ignores `*.epub`, and a fixture
that can be rebuilt from a script can be read as well as run.
"""

import os
import struct
import zipfile
import zlib
from pathlib import Path

from ebooklib import epub

BOOKS_DIR = Path(__file__).resolve().parent / "books"

# The book every real translation in the suite is run against. Short — three
# chapters of a few hundred characters — because every one of them is paid for,
# but written as continuous prose rather than filler: a model given nonsense
# answers with nonsense, and the point is to see a translation that works.
_ZH_CHAPTERS = [
    ("青石镇的铁匠", [
        "青石镇的清晨总是被铁锤声唤醒。天还没有全亮，李明就已经站在炉火前，"
        "手里握着一把还没有开刃的长剑。炉膛里的火苗一跳一跳，把他的影子投在墙上，忽长忽短。",
        "他的师父王秀兰在世的时候常说：“铁是有脾气的。你越着急，它越不肯听你的话。”"
        "那时候李明还小，只觉得这话像绕口令；如今他自己成了镇上唯一的铁匠，才慢慢懂了其中的意思。",
        "那天早上，镇口来了一个背着竹篓的陌生人。他的鞋上沾满了泥，斗笠压得很低，看不清脸。"
        "他把一把断成两截的剑放在柜台上，问李明能不能修好。",
        "李明拿起断剑，指尖在剑身上摸到一朵极小的梅花。那一瞬间，他的手停住了。"
        "十年前那场大火之后，他以为这把寒霜剑早就随着师父一起埋在了后山。",
        "“这把剑从哪里来的？”李明问。陌生人没有回答，只是从竹篓里取出一个布包，轻轻放在剑的旁边。",
    ]),
    ("布包里的东西", [
        "布包里是半块玉佩，颜色发暗，边缘有一道很深的裂痕。李明认得它——另外半块，"
        "这些年一直挂在他自己的脖子上。",
        "“王秀兰是我的姐姐。”陌生人终于开口，声音沙哑得像被烟熏过，“她把剑留给你，把玉佩留给我。"
        "她说，等这两样东西再见面的时候，青石镇就该有麻烦了。”",
        "李明沉默了很久。炉子里的火渐渐弱下去，屋子里的温度一点点降了下来。"
        "他忽然想起师父临走前的那个夜晚：她坐在院子里磨刀，一直磨到天亮，却始终没有说为什么。",
        "“麻烦已经来了。”陌生人说，“昨天夜里，山那边的三个村子都空了。没有打斗的痕迹，"
        "没有血，人就那么不见了。”",
        "李明把断剑放回柜台，转身从墙上取下自己的锤子。“修剑要三天。”他说，“但是我可以先跟你去看看。”",
    ]),
    ("山谷里的回声", [
        "他们在天黑之前赶到了第一个空村子。屋门都开着，锅里的粥还是温的，桌上的碗筷摆得整整齐齐，"
        "好像所有人只是出去了一小会儿。",
        "李明蹲下身，在门槛边发现了一道细细的划痕，很新，边缘还发着白。他用手指量了量，"
        "那宽度和寒霜剑的剑身几乎一模一样。",
        "“你师父当年也在这里站过。”陌生人指着院子中央那口井，“她说这口井底下有回声，可是那时候我不信。”",
        "李明走到井边，把耳朵贴在冰凉的石沿上。起初什么也没有；过了一会儿，他听见了——"
        "很轻，很远，像有人在很深的地方一下一下地敲着铁。",
        "那声音的节奏，正是师父教他的第一段打铁的调子。",
    ]),
]

# Already in English, and meant to stay that way: uploaded as part of a Chinese
# book, these are what the script check has to recognise and leave alone.
_EN_CHAPTERS = [
    ("A Letter from the Capital", [
        "The letter arrived three days after the first village emptied, carried by a "
        "rider who would not come inside and would not give his name.",
        "It bore no seal. The paper was expensive, the hand was careful, and every "
        "sentence in it had been written by somebody who expected to be obeyed.",
        "Whoever wrote it knew about the well. That was the part the blacksmith kept "
        "coming back to, long after the rider had gone.",
    ]),
    ("What the Magistrate Knew", [
        "The magistrate had held his post for eleven years, and in that time he had "
        "learned exactly how much of the truth a report could carry before it became "
        "somebody else's problem.",
        "He read the letter twice, folded it, and put it in the drawer where he kept "
        "the things he intended to forget.",
        "Outside, the rain had started again, and the road to the mountains was "
        "already turning to mud.",
    ]),
]


def _png_pixel() -> bytes:
    """
    A one-pixel PNG, built rather than pasted, so the cover is real image data
    the server can serve with a media type it inferred for itself.
    """
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)   # 1x1, 8-bit truecolour
    pixel = zlib.compress(b"\x00\xc8\x64\x32")              # filter byte, then one RGB pixel

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", pixel)
            + chunk(b"IEND", b""))


def _document(uid: str, title: str, paragraphs, lang: str) -> epub.EpubHtml:
    body = "\n".join(f"    <p>{text}</p>" for text in paragraphs)
    item = epub.EpubHtml(uid=uid, title=title, file_name=f"{uid}.xhtml", lang=lang)
    # No XML declaration: ebooklib parses this string with lxml when it builds the
    # nav document, and lxml refuses a str that declares its own encoding.
    item.content = (
        f"<html xmlns=\"http://www.w3.org/1999/xhtml\">\n"
        f"  <head><title>{title}</title></head>\n"
        f"  <body>\n    <h1>{title}</h1>\n{body}\n  </body>\n</html>\n"
    )
    return item


def _build(path: Path, *, title, author, language, chapters,
           cover: bool = True, padding_bytes: int = 0) -> Path:
    """
    Write one EPUB, the way most of them arrive: an NCX and a nav document
    already in place, which is the case the writer must not add a second set to.
    """
    book = epub.EpubBook()
    book.set_identifier(f"urn:uuid:test-{path.stem}")
    book.set_title(title)
    book.set_language(language)
    if author:
        book.add_author(author)

    items = []

    if cover:
        # The page is written here rather than left to `set_cover(create_page=True)`,
        # which builds a document ebooklib's own nav generator then fails to parse.
        # It holds an image and no words, which is the point: the reader must not
        # offer it as a chapter to translate.
        book.set_cover("cover.png", _png_pixel(), create_page=False)
        page = epub.EpubHtml(uid="cover_page", title="Cover", file_name="cover_page.xhtml")
        page.content = (
            "<html xmlns=\"http://www.w3.org/1999/xhtml\"><head><title>Cover</title></head>"
            "<body><div><img src=\"cover.png\" alt=\"\"/></div></body></html>"
        )
        book.add_item(page)
        items.append(page)

    for index, (chapter_title, paragraphs, chapter_lang) in enumerate(chapters, 1):
        item = _document(f"chap_{index:02d}", chapter_title, paragraphs, chapter_lang)
        book.add_item(item)
        items.append(item)

    if padding_bytes:
        # Random, because an EPUB is a zip: a megabyte of zeroes would compress
        # away and the upload limit would never see it.
        book.add_item(epub.EpubItem(uid="padding", file_name="padding.bin",
                                    media_type="application/octet-stream",
                                    content=os.urandom(padding_bytes)))

    book.toc = [epub.Link(item.file_name, item.title, item.id) for item in items]
    book.spine = ["nav"] + items
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(path), book, {})
    return path


_CONTAINER_XML = """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""


def _build_by_hand(path: Path, *, language: str, chapters) -> Path:
    """
    An EPUB with no NCX, no nav document, no cover and no title, written as a
    zip rather than through ebooklib.

    It has to be built by hand: ebooklib's writer always emits `<spine toc="ncx">`
    whether or not the book has an NCX, and reading that back looks for a
    manifest entry that isn't there and raises. A real EPUB3 without one simply
    leaves the attribute off, which is what this does — and which is the case
    `EpubWriter._ensure_navigation` exists to repair.
    """
    documents = {
        f"chap_{index:02d}.xhtml": (
            f"<html xmlns=\"http://www.w3.org/1999/xhtml\">\n"
            f"  <head><title>{title}</title></head>\n"
            f"  <body>\n    <h1>{title}</h1>\n"
            + "\n".join(f"    <p>{text}</p>" for text in paragraphs)
            + "\n  </body>\n</html>\n"
        )
        for index, (title, paragraphs, _lang) in enumerate(chapters, 1)
    }

    manifest = "\n".join(
        f'    <item id="{Path(name).stem}" href="{name}" media-type="application/xhtml+xml"/>'
        for name in documents
    )
    spine = "\n".join(f'    <itemref idref="{Path(name).stem}"/>' for name in documents)

    opf = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="bookid">urn:uuid:test-{path.stem}</dc:identifier>
    <dc:language>{language}</dc:language>
  </metadata>
  <manifest>
{manifest}
  </manifest>
  <spine>
{spine}
  </spine>
</package>
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        # First, uncompressed, and without the usual archive extras: that is what
        # makes a zip an EPUB.
        archive.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip",
                         compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", _CONTAINER_XML)
        archive.writestr("OEBPS/content.opf", opf)
        for name, content in documents.items():
            archive.writestr(f"OEBPS/{name}", content)

    return path


_STYLESHEET = """body { font-family: serif; line-height: 1.6; margin: 1em; }
h1 { text-align: center; }
p.first { text-indent: 0; }
"""


def _build_styled(path: Path) -> Path:
    """
    A book shaped like the ones people actually upload: chapters in their own
    directory, a stylesheet linked from every chapter's head, a `<title>` in
    each head, and an image referenced from the text.

    Built by hand for the same reason as `_build_by_hand`: what is being tested
    is whether the writer gives back what it was given, so what it is given has
    to be written by something other than the library doing the giving back.
    """
    chapters = {
        "ch1.xhtml": ("第一章 · 青石镇的铁匠", _ZH_CHAPTERS[0][1]),
        "ch2.xhtml": ("第二章 · 布包里的东西", _ZH_CHAPTERS[1][1]),
    }

    documents = {}
    for index, (name, (title, paragraphs)) in enumerate(chapters.items()):
        body = "\n".join(f'    <p class="first">{text}</p>' for text in paragraphs)
        picture = '    <div class="plate"><img src="../Images/pic.png" alt="plate"/></div>\n'
        documents[f"Text/{name}"] = (
            f"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            f"<!DOCTYPE html>\n"
            f"<html xmlns=\"http://www.w3.org/1999/xhtml\">\n"
            f"  <head>\n"
            f"    <title>{title}</title>\n"
            f"    <link href=\"../Styles/main.css\" rel=\"stylesheet\" type=\"text/css\"/>\n"
            f"  </head>\n"
            f"  <body>\n    <h1>{title}</h1>\n{picture if index == 0 else ''}{body}\n"
            f"  </body>\n</html>\n"
        )

    navigation = (
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<html xmlns=\"http://www.w3.org/1999/xhtml\" xmlns:epub=\"http://www.idpf.org/2007/ops\">\n"
        "  <head><title>Contents</title></head>\n"
        "  <body><nav epub:type=\"toc\"><ol>\n"
        + "".join(f'    <li><a href="{name}">{title}</a></li>\n'
                  for name, (title, _) in chapters.items())
        + "  </ol></nav></body>\n</html>\n"
    )

    manifest = "\n".join([
        '    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        '    <item id="css" href="Styles/main.css" media-type="text/css"/>',
        '    <item id="pic" href="Images/pic.png" media-type="image/png"/>',
    ] + [
        f'    <item id="{Path(name).stem}" href="{name}" media-type="application/xhtml+xml"/>'
        for name in documents
    ])
    spine = "\n".join(f'    <itemref idref="{Path(name).stem}"/>' for name in documents)

    opf = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="bookid">urn:uuid:test-styled</dc:identifier>
    <dc:title>青石镇的铁匠 (排版版)</dc:title>
    <dc:creator>佚名</dc:creator>
    <dc:language>zh</dc:language>
  </metadata>
  <manifest>
{manifest}
  </manifest>
  <spine>
{spine}
  </spine>
</package>
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip",
                         compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", _CONTAINER_XML)
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr("OEBPS/nav.xhtml", navigation)
        archive.writestr("OEBPS/Styles/main.css", _STYLESHEET)
        archive.writestr("OEBPS/Images/pic.png", _png_pixel())
        for name, content in documents.items():
            archive.writestr(f"OEBPS/{name}", content)

    return path


def _repeated(chapters, count):
    """`count` chapters cycled out of `chapters`, each numbered and named apart."""
    return [
        (f"第{index + 1}章 · {chapters[index % len(chapters)][0]}",
         chapters[index % len(chapters)][1],
         chapters[index % len(chapters)][2])
        for index in range(count)
    ]


def build_all() -> dict:
    """Write every fixture and hand back where each one landed."""
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    zh = [(title, paragraphs, "zh") for title, paragraphs in _ZH_CHAPTERS]
    en = [(title, paragraphs, "en") for title, paragraphs in _EN_CHAPTERS]

    books = {
        # Three Chinese chapters, a cover, full metadata. The workhorse.
        'zh_small': _build(BOOKS_DIR / "zh_small.epub",
                           title="青石镇的铁匠", author="佚名", language="zh",
                           chapters=zh),

        # Chinese and English interleaved: the English ones must be reported as
        # already translated and left out of the default plan.
        'zh_mixed': _build(BOOKS_DIR / "zh_mixed.epub",
                           title="青石镇 (双语)", author="佚名", language="zh",
                           chapters=[zh[0], en[0], zh[2], en[1]], cover=False),

        # No cover, no author, no title, no navigation of any kind.
        'zh_plain': _build_by_hand(BOOKS_DIR / "zh_plain.epub",
                                   language="zh", chapters=[zh[2]]),

        # Long enough that a run can be caught in the middle and cancelled.
        'zh_long': _build(BOOKS_DIR / "zh_long.epub",
                          title="青石镇的铁匠 (全)", author="佚名", language="zh",
                          chapters=_repeated(zh, 8)),

        # Laid out like a book from a shop: a stylesheet, an image, and chapters
        # in their own directory. What the writer has to give back unchanged.
        'styled': _build_styled(BOOKS_DIR / "styled.epub"),

        # Only here to be too big for MAX_UPLOAD_MB.
        'padded': _build(BOOKS_DIR / "padded.epub",
                         title="青石镇的铁匠 (大)", author="佚名", language="zh",
                         chapters=[zh[0]], cover=False, padding_bytes=1_400_000),
    }

    # Not an EPUB at all, and not a zip either: the first is refused on its name,
    # the second gets as far as the parser.
    text_file = BOOKS_DIR / "notes.txt"
    text_file.write_text("This is not a book.\n", encoding="utf-8")
    books['not_an_epub'] = text_file

    corrupt = BOOKS_DIR / "corrupt.epub"
    corrupt.write_bytes(b"PK\x03\x04 this is not a readable archive " + os.urandom(256))
    books['corrupt'] = corrupt

    return books


if __name__ == "__main__":
    for name, path in build_all().items():
        print(f"{name:14} {path.stat().st_size:>9,} bytes  {path}")
