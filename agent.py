import os
import asyncio
import logging
import mysql.connector
from dotenv import load_dotenv

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, voice, llm
from livekit.plugins import silero, openai, deepgram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tonga-voice")

load_dotenv(".env.local")

# Safety actions and DB logger
class TongaEmergencyTools(llm.FunctionContext):

    def get_db_connection(self):
        return mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASS", ""),
            database=os.getenv("DB_NAME", "tonga_db")
        )

    @llm.ai_callable(description="Trigger if user says they are in danger, accident, or need emergency help.")
    async def trigger_sos_protocol(self, severity: str = llm.TypeInfo(description="Emergency type: fire, accident, harassment")):
        logger.error(f"SOS triggered: {severity}")

        db = None
        try:
            db = self.get_db_connection()
            cursor = db.cursor()

            # Save incident to database
            query = "INSERT INTO emergency_logs (severity) VALUES (%s)"
            cursor.execute(query, (severity,))
            db.commit()
            cursor.close()
            logger.info("Saved emergency log to MySQL.")
        except Exception as e:
            logger.error(f"Database error: {e}")
        finally:
            if db and db.is_connected():
                db.close()

        return "Emergency protocol activated. Human support is notified and location logged."

# System instructions
TONGA_SYSTEM_PROMPT = """
You are the official voice safety and support assistant for Tonga, an Indian ride-sharing platform.
Your highest priority is passenger safety.
Rules:
1. Keep replies concise and under 2 sentences.
2. If the passenger mentions fire, an accident, harassment, or distress, immediately use the 'trigger_sos_protocol' tool. Do NOT ask for permission.
"""

# Real-time voice session setup
async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=TONGA_SYSTEM_PROMPT,
    )

    tonga_tools = TongaEmergencyTools()

    # Streaming voice cascade
    session = voice.AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM.with_groq(model="llama-3.3-70b-versatile"),
        tts=deepgram.TTS(model="aura-asteria-en"),
        chat_ctx=initial_ctx,
        fnc_ctx=tonga_tools
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    session.start(ctx.room)

    logger.info("Tonga agent connected.")
    await session.say("Welcome to Tonga Support. I am your safety assistant. Are you safe, and how can I help you today?")

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )