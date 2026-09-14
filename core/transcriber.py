import whisper
import os

whisper_model = os.getenv("WHISPER_MODEL", "small")

_model= None

sarvam_model = os.getenv("SARVAM_API_KEY", "sk_ldj590kw_THmeINulr0AnWyUaxRLWw9j3")

def load_model():

  global _model

  if _model is None:
    print(f"loading model ...")
    _model = whisper.load_model(whisper_model)
    print("whisper model loaded sucessfully")

  return _model

def transcribe_chunk(chunk_path: str , translate: bool = False) -> str:
    
    model = load_model()

    task = "translate" if translate else "transcribe"

    result= model.transcribe(chunk_path, task=task)

    return result["text"]

def transcribe_all(chunk: list, translate= bool= False) -> str:
  full_transcript = ""

  for i, chunk in enumerate(chunk):
        print(f"transcribing chunk {i+1}")
        text = transcribe_chunk(chunk, translate=translate)

        full_transcript += text + " "

        return full_transcript.strip()

   
