from src.kokoro_tts_engine import synthesize_response_to_wav

audio_path = synthesize_response_to_wav(
    "Bonjour, je suis l'assistant vocal du dashboard. Les ventes sont plus fortes dans la région Nord."
)

print(audio_path)