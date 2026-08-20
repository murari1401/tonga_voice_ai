import os
import asyncio
import logging
from dotenv import load_dotenv

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, voice, llm
from livekit.plugins import silero, openai, deepgram

# Setting up logging so we can see errors in the terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tonga-voice")

load_dotenv(".env.local")

#Function Calling
class TongaEmergencyTools(llm.FunctionContext):
    """
    actions the AI take when user speaks:
    """

    @llm.ai_callable(description="Trigger if user says they are in danger, accident, need police.")
    async def trigger_sos_protocol(self, severity: str = llm.TypeInfo(description="Types: fire, accident, harassment")):
        # When the AI triggers this, it stops generating text and runs this Python code!
        logger.error(f" CRITICAL SOS TRIGGERED Type: {severity}")

        # In the future, here  we will write the code to text the police
        # and send the GPS coordinates to our database.

        return "Emergency protocol activated. Human support is being connected."

#System Prompt
TONGA_SYSTEM_PROMPT = """
You are the official voice safety and support assistant for Tonga, an Indian ride-sharing platform.
Your highest priority is passenger safety.
Rules:
1. Keep replies concise and under 2 sentences.
2. If the passenger mentions fire, an accident, harassment, or distress, immediately use the 'trigger_sos_protocol' tool. Do NOT ask for permission.
"""

#cascade voice session
async def entrypoint(ctx: JobContext):

    #Create the system context with our prompt and tools
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=TONGA_SYSTEM_PROMPT,
    )

    tonga_tools = TongaEmergencyTools()

    #Build the Streaming Cascade (Ears, Brain, Mouth)
    logger.info("Building the Tonga AI Brain...")
    session = voice.AgentSession(
        vad=silero.VAD.load(),

        # EARS
        stt=deepgram.STT(),

        # BRAIN
        llm=openai.LLM.with_groq(model="llama-3.3-70b-versatile"),

        # MOUTH
        tts=deepgram.TTS(model="aura-asteria-en"),

        # MEMORY & ACTIONS
        chat_ctx=initial_ctx,
        fnc_ctx=tonga_tools
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    session.start(ctx.room)

    logger.info("Tonga AI is online and listening!")
    await session.say("Welcome to Tonga Support. I am your safety assistant. Are you safe, and how can I help you today?")

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )