from queue import Queue

import telebot
from langchain.embeddings import HuggingFaceEmbeddings

import config
from src.llm import LLAMA_QA
from src.retriever import ChromaRetriever
from src.worker import WorkerThread

if __name__ == "__main__":

    query_queue = Queue()
    bot = telebot.TeleBot(config.TOKEN)


    @bot.message_handler(commands=["start"])
    def start(message):
        bot.send_message(message.chat.id, config.WELCOME_MESSAGE)


    @bot.message_handler(commands=["help"])
    def help(message):
        bot.send_message(message.chat.id, config.HELP_MESSAGE)


    @bot.message_handler(commands=["info"])
    def info(message):
        bot.send_message(message.chat.id, config.INFO_MESSAGE)


    @bot.message_handler(func=lambda m: True)
    def echo(message):
        if query_queue.qsize() > config.MAX_QUEUE:
            bot.send_message(
                message.chat.id,
                "Извините, ваш вопрос не был добавлен в очередь из-за слишком большого количества других вопросов. Попробуйте позже",
            )
            return

        query_queue.put(message)
        bot.send_message(message.chat.id, "Ваш вопрос добавлен в очередь!")


    embeddings_model = HuggingFaceEmbeddings(
        model_name=config.EMBEDDER_NAME, model_kwargs={"device": "cpu"}
    )
    retriever = ChromaRetriever(embeddings_model)

    qa_model = LLAMA_QA()
    queue_handler = WorkerThread(query_queue, bot, qa_model, retriever)
    queue_handler.start()

    while True:
        try:
            bot.polling()
        except ConnectionError as e:
            print('Connection error ' + str(e))
