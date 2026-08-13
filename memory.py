from dotenv import load_dotenv
import tiktoken
from openai import OpenAI

load_dotenv()

client = OpenAI()


class ConversationMemory:

     def __init__(self, max_tokens=4000):
        self.max_tokens = max_tokens

        self.encoding = tiktoken.get_encoding(
            "cl100k_base"
        )

        self.summary = ""

        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are Alex, a helpful voice assistant. "
                    "Keep your answers concise and natural "
                    "for spoken conversation."
                )
            }
        ]

     def add_user_message(self, text):
        self.messages.append({
            "role": "user",
            "content": text
        })

     def add_assistant_message(self, text):
        self.messages.append({
            "role": "assistant",
            "content": text
        })

     def count_tokenss(self):
        total = 0

        for message in self.messages:
            print(message)
            total += len(
                self.encoding.encode(
                    message["content"]
                )
            )

        return total
     
     def count_tokens(self):
            total = 0

            for message in self.messages:

                content = message.get("content")

                if isinstance(content, str):
                    total += len(
                        self.encoding.encode(content)
                    )

            return total

     def get_messages(self):
        return self.messages

     def need_summary(self):
        if self.count_tokens() > self.max_tokens:
           return True

     def summarize(self):
        response = client.chat.completions.create(
           model="gpt-4o-mini",
           messages=[
              {
                 "role": "system",
                 "content": "You are supposed to summarize the previous messages of user and assistant conversation and pass on only which things are important topic, context only."
              },
              {
                 "role": "system",
                 "content": f"Previous messages: \n {self.messages}"
              }
           ]
        )

        self.summary = response.choices[0].message.content

     def compress(self):

        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are Alex, a helpful voice assistant. "
                    "Keep your answers concise and natural "
                    "for spoken conversation."
                )
            },
            {
                "role": "system",
                "content": (
                    f"Previous conversation summary:\n"
                    f"{self.summary}"
                )
            }
        ]