import logging

import certifi
import urllib3
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
import ask_sdk_core.utils as ask_utils
from ask_sdk_core.handler_input import HandlerInput

from ask_sdk_model import Response
import os
import openai

PERSONAS = {
    "assistant": (
        "I am a highly intelligent question answering bot. If you ask me a "
        "question that is rooted in truth, I will give you the answer. If you "
        "ask me a question that is nonsense, trickery, or has no clear answer, I "
        "will respond with a humorous nonsense response. My response will be "
        "converted to speech, so I will not respond with tables or other "
        "information that cannot be verbalized. If I'm asked specifically for "
        "information that cannot be verbalised, I will reply explaining that the "
        "information is too difficult to express verbally and they should re-ask "
        "when they are in front of a screen."
    ),
    "kpop": (
        "I am a high school girl who is obsessed with KPOP and will respond as such. "
        "I will respond as if I were talking with my high school friends I will use "
        "slang and abbreviations that are common in KPOP stan culture. I will not be "
        "overly formal. I will not use emojis or emoticons. I will never ask questions "
        "of the user.  I will double check all my responses to ensure I am not asking "
        "a quesiton and that I am not using emojis or emoticons. "
        "Unless otherwise instructed, I will assume your question is related to KPOP "
        "and even if it is not, I'll likely make a KPOP music reference in my response."
        "My bias is Jung Kook and I think he is just the greatest, but I am subtle "
        "it and while I love BTS and Jung Kook, I will eagerly talk about other bands."
        "My responses will be converted to speech, so I will not respond with tables, "
        "or other information that cannot be verbalized. If I'm asked specifically for "
        "information that cannot be verbalised nor will I respond with questions. I "
        "will reply explaining that the information is too difficult to express "
        "verbally and they should re-ask when they are in front of a screen."
    ),
}

# OPEN AI Config
# openai.organization = os.getenv("OPENAI_API_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")
# model = os.getenv("MODEL", "text-davinci-003")
model = os.getenv("MODEL", "gpt-3.5-turbo")
temperature = float(os.getenv("TEMPERATURE", 0.1))
max_tokens = int(os.getenv("MAX_TOKENS", 3000))

# SLACK CONFIG
slack_url = os.getenv("SLACK_URL")
channel = os.getenv("SLACK_CHANNEL", "#chatgpt")

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGLEVEL", logging.DEBUG))

RE_PROMPT = "Do you have any other questions?"


# -- ChatGPT Functions --
def get_chat_response(messages):
    logger.debug(f"messages for OpenAI: {messages}")
    # create variables to collect the stream of chunks
    collected_chunks = []
    collected_messages = []
    for chunk in openai.ChatCompletion.create(
        model=model, messages=messages, stream=True
    ):
        collected_chunks.append(chunk)  # save the event response
        try:
            chunk_message = chunk["choices"][0]["delta"]["content"]
        except KeyError:
            stop_reason = chunk["choices"][0]["finish_reason"]
            logger.debug(f"get_chat_response stop_reason: {stop_reason}")
        else:
            collected_messages.append(chunk_message)  # save the message
            # print(chunk_message, end="", flush=True)
    response = "".join(collected_messages)
    logger.debug(f"response: {response}")
    return response


class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        # Get any existing attributes from the incoming request
        session_attr = handler_input.attributes_manager.session_attributes

        # check to see if there is a personal already set, if so, leave it.
        if "0" not in session_attr:
            session_attr["0"] = {"role": "system", "content": PERSONAS["assistant"]}

        speak_output = "ChatGPT here, say help to learn more"
        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )


class ChatGPTIntentHandler(AbstractRequestHandler):
    """Handler for ChatGPTIntent. Must be evaluated after Slack Intent"""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        logger.debug(f"request: {handler_input.request_envelope.request}")
        return ask_utils.is_request_type("IntentRequest")(
            handler_input
        ) and ask_utils.get_intent_name(handler_input).startswith("ChatGPT")

    def handle(self, handler_input: HandlerInput) -> Response:
        question = get_question(handler_input)
        logger.debug(
            f"intent: {ask_utils.get_intent_name(handler_input)}\nquestion: {question}"
        )

        # Get any existing attributes from the incoming request
        session_attr = handler_input.attributes_manager.session_attributes
        logger.debug(f"session_attr: {session_attr}")

        # I'm storing the chatgpt sesion in the session_attributes; chatgpt need them
        # to be in order, so I'm using a dictionary with the index as the key.  The
        # session_atr will already have the persona at '0', so I'm starting the index at 1

        max_key = int(max(session_attr)) + 1

        messages = []

        # personal always goes first
        messages.append(session_attr["0"])

        # Add any existing messages to the prompt
        for i in range(1, max_key):
            messages.append(session_attr[str(i)])

        # Add the user's question to the prompt
        messages.append({"role": "user", "content": question})

        # Get the response from ChatGPT
        speak_output = get_chat_response(messages)

        # Add the response to the prompt
        messages.append({"role": "system", "content": speak_output})

        # Save the messages in the session attributes
        key_index = 0
        for message in messages:
            session_attr[str(key_index)] = message
            key_index += 1

        logging.debug(f"session_attr: {session_attr}")

        return (
            handler_input.response_builder.speak(speak_output)
            .ask(RE_PROMPT)
            .set_should_end_session(False)
            .response
        )


def get_question(handler_input: HandlerInput) -> str:
    request = handler_input.request_envelope.request
    # Hack to capture the first trigger word by extracting from the intent name
    # Example ChatGPTDefineIntent will return Define below which is the initial trigger
    # for this request
    first_word = ask_utils.get_intent_name(handler_input).split("ChatGPT")[1][:-6]
    return first_word + " " + request.intent.slots["question"].value


class ImageHandler(AbstractRequestHandler):
    """Handler for ImageHandler."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_request_type("IntentRequest")(
            handler_input
        ) and ask_utils.is_intent_name("ImageHandler")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        question = handler_input.request_envelope.request.intent.slots["question"].value

        image_url = openai.Image.create(
            prompt=question, n=1, size="1024x1024", response_format="url"
        ).data[0]["url"]

        res = send_slack_message(question=question, image_url=image_url)
        return (
            handler_input.response_builder.speak(f"{res} sending to slack")
            .ask(RE_PROMPT)
            .set_should_end_session(False)
            .response
        )


class ChatGPTSlackHandler(AbstractRequestHandler):
    """Handler for ChatGPTSlackHandler."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_request_type("IntentRequest")(
            handler_input
        ) and ask_utils.is_intent_name("ChatGPTSlackHandler")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        question = handler_input.request_envelope.request.intent.slots["question"].value
        # Remove the first word here as it likely matches a persons name e.g.
        # 'Slack me', 'Message Joe ...'
        question_without_first_word = " ".join(question.split(" ")[1:])
        chatgpt_output = (
            openai.Completion.create(
                model=model,
                prompt=question_without_first_word,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            .choices[0]
            .text
        )

        res = send_slack_message(
            question=question_without_first_word, response=chatgpt_output
        )
        return (
            handler_input.response_builder.speak(f"{res} sending to slack")
            .ask(RE_PROMPT)
            .set_should_end_session(False)
            .response
        )


def send_slack_message(question, response=None, image_url=None) -> str:
    data = {
        "channel": channel,
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": question.capitalize()},
            },
            {"type": "divider"},
        ],
    }

    if response:
        data["blocks"].append(
            {"type": "section", "text": {"type": "mrkdwn", "text": response}}
        )

    if image_url:
        data["blocks"].append(
            {"type": "image", "image_url": image_url, "alt_text": "Haunted hotel image"}
        )

        http = urllib3.PoolManager(cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())

        resp = http.request(
            "POST", slack_url, json=data, headers={"Content-Type": "application/json"}
        )

        if resp.status != 200:
            logger.error(resp.status, resp.data)
            return "failed"
    return "success"


class HelpIntentHandler(AbstractRequestHandler):
    """Handler for Help Intent."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_request_type("IntentRequest")(
            handler_input
        ) and ask_utils.is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        speak_output = "You can say hello to me! How can I help?"
        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )


class NewSessionIntentHandler(AbstractRequestHandler):
    """Handler to reset sesion."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_request_type("IntentRequest")(
            handler_input
        ) and ask_utils.is_intent_name("NewSessionIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        # Get any existing attributes from the incoming request
        session_attr = handler_input.attributes_manager.session_attributes

        logger.debug(f"Old Session Attributes {session_attr}")
        # remove all session attributes except for ["0"] which stores the persona
        current_persona = session_attr["0"]
        session_attr.clear()
        session_attr["0"] = current_persona

        logger.debug(f"New Session Attributes {session_attr}")

        speak_output = "You can say hello to me! How can I help?"
        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )


class ChangePersonaHandler(AbstractRequestHandler):
    """Change the chatGPT Persona"""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        logger.debug(f"request: {handler_input.request_envelope.request}")
        return ask_utils.is_request_type("IntentRequest")(
            handler_input
        ) and ask_utils.get_intent_name(handler_input).startswith("Persona")

    def handle(self, handler_input: HandlerInput) -> Response:
        """Change persona based on the intent name"""

        session_attr = handler_input.attributes_manager.session_attributes

        intent_name = ask_utils.get_intent_name(handler_input)
        # Get the persona which will be after "Persona" and before "Intent"
        persona_name = intent_name.split("Persona")[1].split("Intent")[0].lower()

        if persona_name not in PERSONAS:
            speak_output = f"Sorry, I don't know the persona {persona_name}"
            return (
                handler_input.response_builder.speak(speak_output)
                .ask(speak_output)
                .response
            )
        session_attr["0"] = {"role": "system", "content": PERSONAS[persona_name]}
        speak_output = f"Persona changed to {persona_name}"
        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )


class CancelOrStopIntentHandler(AbstractRequestHandler):
    """Single handler for Cancel and Stop Intent."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_request_type("IntentRequest")(
            handler_input
        ) and ask_utils.get_intent_name(handler_input) in [
            "AMAZON.StopIntent",
            "AMAZON.CancelIntent",
        ]

    def handle(self, handler_input: HandlerInput) -> Response:
        return (
            handler_input.response_builder.set_should_end_session(True)
            .speak("OK, ChatGPT out")
            .response
        )


class SessionEndedRequestHandler(AbstractRequestHandler):
    """Handler for Session End."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        # Any cleanup logic goes here.
        logger.debug("SessionEndedRequest receieved")
        return handler_input.response_builder.response


class IntentReflectorHandler(AbstractRequestHandler):
    """The intent reflector is used for interaction model testing and debugging.
    It will simply repeat the intent the user said. You can create custom handlers
    for your intents by defining them above, then also adding them to the request
    handler chain below.
    """

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_request_type("IntentRequest")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        intent_name = ask_utils.get_intent_name(handler_input)
        speak_output = "You just triggered " + intent_name + "."
        return handler_input.response_builder.speak(speak_output).response


class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Generic error handling to capture any syntax or routing errors. If you receive
    an error stating the request handler chain is not found, you have not implemented
    a handler for the intent being invoked or included it in the skill builder below.
    """

    def can_handle(self, handler_input: HandlerInput, exception: Exception) -> bool:
        return True

    def handle(self, handler_input: HandlerInput, exception: Exception) -> Response:
        logger.error(exception, exc_info=True)
        speak_output = "Sorry, I had trouble doing what you asked. Please try again."

        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )


# The SkillBuilder object acts as the entry point for your skill, routing all request
# and response payloads to the handlers above. Make sure any new handlers or
# interceptors you've defined are included below. The order matters - they're processed
# top to bottom.

sb = SkillBuilder()
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(ChatGPTSlackHandler())
sb.add_request_handler(ImageHandler())
sb.add_request_handler(ChatGPTIntentHandler())
sb.add_request_handler(NewSessionIntentHandler())
sb.add_request_handler(ChangePersonaHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
# make sure IntentReflectorHandler is last so it doesn't override your custom intent
# handlers
sb.add_request_handler(IntentReflectorHandler())

sb.add_exception_handler(CatchAllExceptionHandler())
handler = sb.lambda_handler()
