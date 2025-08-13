import os
import logging
import datetime
import sqlite3
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters
from dotenv import load_dotenv
from rag_engine import find_similar_chunks
from openai import OpenAI

# Load environment variables
load_dotenv()
TELEGRAM_API_KEY = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()
client.api_key = OPENAI_API_KEY

# System prompt
SYSTEM_PROMPT = """Ты — заботливый, умный и харизматичный медицинский ассистент команды Nikolife. 
Ты эксперт по метаболическому здоровью, снижению веса, питанию, гормонам и БАДам.

💡 Правила работы:
1️⃣ Если ответ есть в нашей базе данных — выдай полный, максимально детальный ответ, используя абсолютно все найденные факты.
2️⃣ Обязательно указывай точные цифры, дозировки, временные интервалы, механизмы действия (например: "150 минут умеренной активности в неделю", "25–30 г клетчатки в день").
3️⃣ Структурируй ответ по пунктам, добавляй пошаговые рекомендации, чтобы человек знал, что делать прямо сегодня.
4️⃣ Будь тёплым, мотивирующим, дружелюбным 💖 — используй уместные эмодзи, чтобы текст был живым и поддерживающим.
5️⃣ Если речь о БАДах или витаминах — укажи дозировку, пользу, риски и противопоказания.
6️⃣ Никогда не ставь диагноз и не назначай лечение — напоминай, что при серьёзных симптомах нужно обратиться к врачу.

📌 Формат при наличии данных в базе:
Начни с приветствия: "🌿 Спасибо за вопрос и добро пожаловать в семью Nikolife!"
Дай развернутый, насыщенный фактами и цифрами ответ, используй уместные эмодзи (7–10). Пусть ответы будут длинными, но подробными, и формулируй их в виде приказов, например, вам следует выходить на прогулку в течение 30 минут каждый день. Используйте мотивирующие фразы и ведите себя как лучший друг.

🚀 Если в базе данных нет информации:
Скажи: " 🌿 Спасибо за вопрос и добро пожаловать в семью Nikolife! 😔 К сожалению, в нашей базе нет информации по вашему запросу… но я — умный ассистент, и вот что я могу рассказать!"
Дай яркий, полезный и запоминающийся ответ с эмодзи и лёгким юмором.

✨ Твоя цель:
Вдохновить, обучить, поддержать и дать чёткий, практичный план действий.
"""

logging.basicConfig(level=logging.INFO)

DB_FILE = "logs.db"

# --- Database functions ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            time TEXT,
            user_id INTEGER,
            question TEXT,
            response TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_log(user_id, question, response):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO logs VALUES (?, ?, ?, ?)",
              (str(datetime.datetime.now()), user_id, question, response))
    conn.commit()
    conn.close()

def get_user_history(user_id, limit=5):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT question, response FROM logs WHERE user_id=? ORDER BY time DESC LIMIT ?",
              (user_id, limit))
    rows = c.fetchall()
    conn.close()
    history_texts = []
    for q, r in reversed(rows):
        history_texts.append(f"User: {q}\nAssistant: {r}")
    return "\n".join(history_texts)

# --- Conversation memory ---
user_conversations = {}

async def ask_openai(user_id, user_prompt, context_chunks=None):
    if user_id not in user_conversations:
        user_conversations[user_id] = []

    if context_chunks:
        context_text = "\n".join(context_chunks)
        user_conversations[user_id].append(
            {"role": "system", "content": f"Контекст из базы данных:\n{context_text}"}
        )

    past_interactions = get_user_history(user_id, limit=3)
    if past_interactions:
        user_conversations[user_id].append(
            {"role": "system", "content": f"Личная история пользователя:\n{past_interactions}"}
        )

    user_conversations[user_id].append({"role": "user", "content": user_prompt})
    recent_history = user_conversations[user_id][-10:]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + recent_history,
        temperature=0.5,
        max_tokens=800
    )

    bot_reply = response.choices[0].message.content
    user_conversations[user_id].append({"role": "assistant", "content": bot_reply})
    return bot_reply

# --- Telegram handlers ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    matched_chunks = find_similar_chunks(user_input, k=5)
    used_knowledge_base = bool(matched_chunks)

    response = await ask_openai(user_id, user_input, matched_chunks if used_knowledge_base else None)
    save_log(user_id, user_input, response)

    await context.bot.send_message(chat_id=chat_id, text=response)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "🌿 *Добро пожаловать в Nikolife AI Assistant!* 💖\n\n"
        "Я — твой умный и заботливый помощник по здоровью, питанию и долголетию. "
        "Вместе мы создадим план, который поможет тебе чувствовать себя лучше, сильнее и счастливее 🌞\n\n"
        "📌 Вот что я могу для тебя сделать:\n"
        "• Подобрать персональные рекомендации по питанию 🥗\n"
        "• Рассказать, как улучшить сон и энергию 😴⚡\n"
        "• Подсказать дозировки витаминов и БАДов 💊\n"
        "• Дать советы по тренировкам и восстановлению 🏋️‍♂️\n\n"
        "💬 Просто напиши свой вопрос — и мы начнём!"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

# --- Main ---
if __name__ == "__main__":
    init_db()
    app = ApplicationBuilder().token(TELEGRAM_API_KEY).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("✅ Bot is running via long polling...")
    app.run_polling()
