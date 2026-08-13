import json
import ast
import time
import operator
from playsound3 import playsound
from dotenv import load_dotenv
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from memory import ConversationMemory
from datetime import datetime
from wake_word import WakeWordDetector
from recorder import record_until_silence


load_dotenv()

wake_detector = WakeWordDetector()

client = OpenAI()

memory = ConversationMemory(max_tokens=400)

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}

def get_current_time():
    return datetime.now().strftime("%I:%M %p")

def calculator(exp):

    try:
        node = ast.parse(exp, mode="eval").body

        def calculate(node):

            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value

                raise ValueError("Invalid value")

            if isinstance(node, ast.BinOp):
                operation = OPERATORS.get(type(node.op))

                if operation is None:
                    raise ValueError("Unsupported operator")

                return operation(
                    calculate(node.left),
                    calculate(node.right)
                )

            if isinstance(node, ast.UnaryOp):
                operation = OPERATORS.get(type(node.op))

                if operation is None:
                    raise ValueError("Unsupported operator")

                return operation(
                    calculate(node.operand)
                )

            raise ValueError("Invalid expression")

        return str(calculate(node))

    except Exception:
        return "I couldn't calculate that."


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate any mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "exp": {
                        "type": "string",
                        "description": "Any mathematical expression such as 25 * 7 / 9"
                    }
                },
                "required": ["exp"]
            }
        }
    }
]

while True:
    try:
        # 1. Wait for "Hey Jarvis"
        wake_detector.wait_for_wake_word()

        # 2. Record until silence
        audio_file = record_until_silence()
    except Exception as e:

        print(f"❌ Audio error: {e}")
        print("🔄 Restarting microphone...")
        time.sleep(1)

        continue

    # 3. STT
    print("Transcribing ✍️")

    try:
        with open(audio_file, 'rb') as audio_file:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_file,
                prompt="""This audio switches back and forth between Bengali, English and hindi (No other language). Transcribe verbatim. 
                
                Any of these following messages - 'exit.', 'quit.', 'bye.' - do translate in english only.
                """   
            )

        text = transcription.text
    except (APIError, APIConnectionError) as e:
        print(f"⚔️ Transcription failed: {e}")
        print("🔀 Returning to wake word..")
        continue

    print(f'You: \n\n {text}')

    if text.lower().strip() in ['exit.', 'quit.', 'bye.']:
        print('Goodbye')
        break

    memory.add_user_message(text)

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=memory.get_messages(),
            tools=tools
        )

    except RateLimitError:

        print("⚠️ API rate limit reached.")
        print("🔄 Returning to wake word...")
        continue

    except APIConnectionError:

        print("⚠️ Could not connect to OpenAI.")
        print("🔄 Returning to wake word...")
        continue

    except APIError as e:

        print(f"❌ OpenAI API error: {e}")
        print("🔄 Returning to wake word...")
        continue

    message = response.choices[0].message

    while message.tool_calls:

        memory.messages.append(message.model_dump())

        for tool in message.tool_calls:

            if tool.function.name == "get_current_time":
                print("Using current time tool")
                result = get_current_time()

            elif tool.function.name == "calculator":
                try:
                    arguments = json.loads(tool.function.arguments)
                    exp = arguments["exp"]
                    print("Using calculator tool")
                    result = calculator(exp)
                except (json.JSONDecodeError, KeyError) as e:

                    print(f"❌ Calculator arguments invalid: {e}")

                    result = "I couldn't understand the mathematical expression."

            else:
                result = f"Unknown tool: {tool.function.name}"

            memory.messages.append({
                "role": "tool",
                "tool_call_id": tool.id,
                "content": result
            })

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=memory.get_messages(),
            tools=tools
        )

        message = response.choices[0].message

    answer = message.content

    if not answer:
        print("No response generated")
        continue

    print(f'Alex: \n\n {answer}')

    memory.add_assistant_message(answer)

    try:
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="sage",
            input=answer
        ) as response:
            response.stream_to_file('response.mp3')
            print('🔊 Response saved')

        playsound('response.mp3')

    except (APIConnectionError, APIError) as e:

        print(f"❌ TTS failed: {e}")

    token_count = memory.count_tokens()

    print(f"Token spend: {token_count}")

    if memory.need_summary():
        print("🧠 Memory limit reached. Summarizing...")

        memory.summarize()
        memory.compress()

        print("✅ Memory compressed.")

