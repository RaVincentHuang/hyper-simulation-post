CONTROLS = {
    '\u0001', '\u0002', '\u0003', '\u0004', '\u0005', '\u0006', '\u0007', '\u0008', '\u000e', '\u000f', '\u0011',
    '\u0012', '\u0013', '\u0014', '\u0015', '\u0016', '\u0017', '\u0018', '\u0019', '\u001a', '\u001b',
}
HYPHENS = {
    '-',
    '‐',
    '‑',
    '⁃',
    '‒',
    '–',
    '—',
    '―',
}
MINUSES = {
    '-',
    '−',
    '－',
    '⁻',
}
PLUSES = {
    '+',
    '＋',
    '⁺',
}
SLASHES = {
    '/',
    '⁄',
    '∕',
}
TILDES = {
    '~',
    '˜',
    '⁓',
    '∼',
    '∽',
    '∿',
    '〜',
    '～',
}
APOSTROPHES = {
    "'",
    '’',
    '՚',
    'Ꞌ',
    'ꞌ',
    '＇',
}
SINGLE_QUOTES = {
    "'",
    '‘',
    '’',
    '‚',
    '‛',
}
DOUBLE_QUOTES = {
    '"',
    '“',
    '”',
    '„',
    '‟',
}
ACCENTS = {
    '`',
    '´',
}
PRIMES = {
    '′',
    '″',
    '‴',
    '‵',
    '‶',
    '‷',
    '⁗',
}
QUOTES = APOSTROPHES | SINGLE_QUOTES | DOUBLE_QUOTES | ACCENTS | PRIMES
def normalize(text):
    for control in CONTROLS:
        text = text.replace(control, '')
    text = text.replace('\u000b', ' ').replace('\u000c', ' ').replace(u'\u0085', ' ')
    for hyphen in HYPHENS | MINUSES:
        text = text.replace(hyphen, '-')
    text = text.replace('\u00ad', '')
    for double_quote in DOUBLE_QUOTES:
        text = text.replace(double_quote, '"')
    for single_quote in (SINGLE_QUOTES | APOSTROPHES | ACCENTS):
        text = text.replace(single_quote, "'")
    text = text.replace('′', "'")
    text = text.replace('‵', "'")
    text = text.replace('″', "''")
    text = text.replace('‶', "''")
    text = text.replace('‴', "'''")
    text = text.replace('‷', "'''")
    text = text.replace('⁗', "''''")
    text = text.replace('…', '...').replace(' . . . ', ' ... ')
    for slash in SLASHES:
        text = text.replace(slash, '/')
    return text