import os

SYSTEM_PROMPT = "Ты — русскоязычный автоматический ассистент в Альфа-Банке Беларуси.\
Тебе предложена информация с сайта Альфа-Банка при помощи этой информации нужно подробно отвечать на вопросы клиентов.\
Так же старайся выставлять всю информацию в доброжелательном и позитивном виде.\
В ответах не ссылайся на другие банки и сервисы. Рассказывай только про Альфа-Банк.\
"


# LLAMA2 CONFIG
LLAMA_REPO_NAME= "IlyaGusev/saiga2_7b_ggml"
LLAMA_MODEL_NAME =  "ggml-model-q4_1.bin"
SYSTEM_TOKEN = 1788
USER_TOKEN = 1404
BOT_TOKEN = 9225
LINEBREAK_TOKEN = 13
ROLE_TOKENS = {"user": USER_TOKEN, "bot": BOT_TOKEN, "system": SYSTEM_TOKEN}
MAX_CONTEXT = 4096
LAST_N_TOKENS = 1024
MAX_CONTEXT_WINDOW = 3072
MAX_NEW_TOKENS = 1500
TOP_K = 30
TOP_P = 0.9
TEMP = 0.1
REPEAT_PENALTY = 1.15


# RETRIEVER CONFIG (CHROMA DB)
EMBEDDER_NAME = "intfloat/multilingual-e5-base"
CHROMA_DIR = "./chroma_data"
CHROMA_COLLECTION_NAME = 'texts'
MMR_TOP_K = 30
MMR_LAMBDA = 0.9
TF_TOP_K = 2
SIMILARITY_TOP_K = 3


# BOT SETTINGS
TOKEN = os.environ['CHAT_BOT_TOKEN']
ADMIN_ID = 323147495
MAX_QUEUE = 3


WELCOME_MESSAGE = """
Приветствую!👋
Я - демо чат-бот от Альфа-Банка Беларуси.
Я постараюсь ответить на ваши вопросы по продуктам и услугам банка.
Всю информацию я беру из открытого источника - сайта alfabank.by.
Информация о боте и разработчике: /info
Задайте ваш вопрос!
"""

INFO_MESSAGE = """
Демо чат-бот для ответов на вопросы с использованием информации с сайта alfabank.by.
В бота загружено около 1000 страниц с сайта.
Бот построен по принципу технологии Retrieval QA на базе дообученой под русской язык LLAMA2-7B, квантизованной в 4bit.
В данный момент используется самая простая модель и не поддерживается память в диалоге, это позволяет всему сервису быть запущенным на MacBook Pro M1.
Поэтому ответы бота могут занимать некоторое время и быть не очень корректными.

При улучшени железа можно:
- запустить гораздо более умную модель.
- в разы ускорить время ответа.
- добавить память в диалог.
- загрузить больше информации, в том числе создать базу знаний из ыыццвдокументов.
- дообучить модель на диалогах из чата в приложении.

Разработчик: Ляхнович Кирилл (@Klhnvich)  - сотрудник отдела продвинутой аналитики, Альфа-Банк Беларусь.
Вопросы, предложения, пожелания приветствуются!
"""

HELP_MESSAGE = """
Чтобы задать вопрос, просто напиши его в чат!
Например:
Что такое карта 100 дней?
Можно ли сделать индивидуальный дизайн карты?
Какие условия на взятие в лизинг авто?
"""


