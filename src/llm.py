from huggingface_hub import snapshot_download

import config
from llama_cpp import Llama


class LLAMA_QA:

    def __init__(self, ):
        try:
            self._init_model()
        except Exception as e:
            snapshot_download(repo_id=config.LLAMA_REPO_NAME, local_dir=".", allow_patterns=config.LLAMA_MODEL_NAME)
        finally:
            self._init_model()

    def _init_model(self):
        self.model = Llama(
            model_path=config.LLAMA_MODEL_NAME,
            n_ctx=config.MAX_CONTEXT,
            last_n_tokens_size=config.LAST_N_TOKENS,
            n_parts=1,
            n_gpu_layers=1,
            # n_threads = 8,
            use_mlock=True,
            use_mmap=True,
            low_vram=True,
        )

    def get_system_tokens(self):
        system_message = {"role": "system", "content": config.SYSTEM_PROMPT}
        return self.get_message_tokens(**system_message)

    def get_message_tokens(self, role, content):
        message_tokens = self.model.tokenize(content.encode("utf-8"))
        message_tokens.insert(1, config.ROLE_TOKENS[role])
        message_tokens.insert(2, config.LINEBREAK_TOKEN)
        message_tokens.append(self.model.token_eos())
        return message_tokens

    def generate_answer(self, question, retrieved_texts):
        tokens = self.get_system_tokens()[:]
        tokens.append(config.LINEBREAK_TOKEN)
        prompt = f"Информация с сайта Альфа-Банка Беларуси: {retrieved_texts[:config.MAX_CONTEXT_WINDOW]}\
        \n\n  Используя этот контекст подробно ответь на вопрос: {question}"
        message_tokens = self.get_message_tokens(
            role="user", content=prompt
        )
        tokens.extend(message_tokens)
        role_tokens = [self.model.token_bos(), config.BOT_TOKEN, config.LINEBREAK_TOKEN]
        tokens.extend(role_tokens)
        generator = self.model.generate(
            tokens,
            top_k=config.TOP_K,
            top_p=config.TOP_P,
            temp=config.TEMP,
            repeat_penalty=config.REPEAT_PENALTY,
            reset=True,
        )
        partial_text = ""
        for i, token in enumerate(generator):
            if token == self.model.token_eos() or i >= config.MAX_NEW_TOKENS:
                break
            partial_text += self.model.detokenize([token]).decode("utf-8", "ignore")

        return partial_text
