cyrillic_mapping = {
    'е': 'е',  # Cyrillic small letter e
    'н': 'н',  # Cyrillic small letter en
    'д': 'д',  # Cyrillic small letter de
    'ш': 'ш',  # Cyrillic small letter sha
    'қ': 'қ',  # Cyrillic small letter qa
    'а': 'а',  # Cyrillic small letter a
    'л': 'л',  # Cyrillic small letter el
    'й': 'й',  # Cyrillic small letter short i
    'м': 'м',  # Cyrillic small letter em
    'і': 'і',  # Cyrillic small letter i
    'ж': 'ж',  # Cyrillic small letter zhe
    'ұ': 'ұ',  # Cyrillic small letter straight u
    'ы': 'ы',  # Cyrillic small letter y
    'с': 'с',  # Cyrillic small letter es
    'C': 'с', 
    'о': 'о',  # Cyrillic small letter o
    'ң': 'ң',  # Cyrillic small letter en with descender
    'б': 'б',  # Cyrillic small letter be
    'р': 'р',  # Cyrillic small letter er
    'ү': 'ү',  # Cyrillic small letter straight u with diaeresis
    'т': 'т',  # Cyrillic small letter te
    'п': 'п',  # Cyrillic small letter pe
    'у': 'у',  # Cyrillic small letter u
    'ў': 'у',
    'и': 'и',  # Cyrillic small letter i
    'к': 'к',  # Cyrillic small letter ka
    'э': 'э',  # Cyrillic small letter e
    'ф': 'ф',  # Cyrillic small letter ef
    'ц': 'ц',  # Cyrillic small letter tse
    'ө': 'ө',  # Cyrillic small letter o with diaeresis
    'ә': 'ә',  # Cyrillic small letter schwa
    'ə': 'ә',
    'Ə': 'ә',
    'в': 'в',  # Cyrillic small letter ve
    'ғ': 'ғ',  # Cyrillic small letter ghe with stroke
    'х': 'х',  # Cyrillic small letter ha
    'з': 'з',  # Cyrillic small letter ze
    'г': 'г',  # Cyrillic small letter ge
    'ь': 'ь',  # Cyrillic small letter soft sign
    'я': 'я',  # Cyrillic small letter ya
    'ч': 'ч',  # Cyrillic small letter che
    'ю': 'ю',  # Cyrillic small letter yu
    'ъ': 'ъ',  # Cyrillic small letter hard sign
    'щ': 'щ',  # Cyrillic small letter shcha
    'ё': 'ё',  # Cyrillic small letter yo
}

# kazakh_numerals_ordinal_mapping = {
#     "1": "бірінші",
#     "2": "екінші",
#     "3": "үшінші",
#     "4": "төртінші",
#     "5": "бесінші",
#     "6": "алтыншы",
#     "7": "жетінші",
#     "8": "сегізінші",
#     "9": "тоғызыншы",
#     "10": "оныншы",
#     "11": "он бірінші",
#     "12": "он екінші",
#     "13": "он үшінші",
#     "14": "он төртінші",
#     "15": "он бесінші",
#     "16": "он алтыншы",
#     "17": "он жетінші",
#     "18": "он сегізінші",
#     "19": "он тоғызыншы",
#     "20": "жиырмасыншы",
#     "21": "жиырма бірінші",
#     "22": "жиырма екіінші",
#     "23": "жиырма үшінші",
#     "24": "жиырма төртінші",
#     "25": "жиырма бесінші",
#     "26": "жиырма алтыншы",
#     "27": "жиырма жетінші",
#     "28": "жиырма сегізінші",
#     "29": "жиырма тоғызыншы",
#     "30": "отызыншы",
#     "31": "отыз бірінші"
# }

kazakh_numerals_ordinal_mapping = {
    'бір': 'бірінші',
    'екі': 'екінші',
    'үш': 'үшінші',
    'төрт': 'төртінші',
    'бес': 'бесінші',
    'алты': 'алтыншы',
    'жеті': 'жетінші',
    'сегіз': 'сегізінші',
    'тоғыз': 'тоғызыншы',
    'он': 'оныншы',
    'жиырма': 'жиырмасыншы',
    'отыз': 'отызыншы',
    'қырық': 'қырықыншы',
    'елу': 'елуінші',
    'алпыс': 'алпысыншы',
    'жетпіс': 'жетпісінші',
    'сексен': 'сексенінші',
    'тоқсан': 'тоқсаныншы',
    'жүз': 'жүзінші'
}

kazakh_chars = set(
    'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    'ӘәҒғҚқҢңӨөҰұҮүҺһІі'
    'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    ' ,-.'
    # '0123456789'
)