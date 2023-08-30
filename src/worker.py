import threading
import time
from queue import Queue
import config


class WorkerThread(threading.Thread):
    def __init__(self, query_queue: Queue, bot, qa_model, retriever):
        threading.Thread.__init__(self)
        self.query_queue = query_queue
        self.qa_model = qa_model
        self.bot = bot
        self.retriever = retriever

    def run(self):
        while True:
            message = self.query_queue.get()
            if message is None:
                time.sleep(0.3)
                continue
            self.answer(message)

    def answer(self, message):
        reply = ""
        try:
            user_message = message.text
            retrieved_texts = self.retriever.retrieve_docs(user_message)
            source_url = retrieved_texts.split("\n")[0]
            self.bot.send_message(
                config.ADMIN_ID, f"Вопрос от пользователя {message.chat.id}: {message.text}"
            )

            generated_answer = self.qa_model.generate_answer(user_message, retrieved_texts)
            # append message and source url to reply
            reply = message.text + "\n\n" + generated_answer + "\n" + source_url[10:]

        except Exception as e:
            print(e)
            reply = "Что-то пошло не так. Попробуйте задать другой вопрос или повторить вопрос позже."
        finally:
            try:
                self.bot.send_message(message.chat.id, reply)
                self.bot.send_message(
                    config.ADMIN_ID, f"Ответ для пользователя {message.chat.id}:\n\n{reply}\n"
                )
            except ConnectionError as e:
                print("Cannot send reply")
