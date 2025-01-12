import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForCausalLM
import os
import jiwer

# Determines which device is available and what data type to use for calculations.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Transcribes given audio file
def transcribe():
    # Specifies to use whisper large v3 turbo model.
    model_id = "openai/whisper-large-v3-turbo"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    # Sets up the pipeline with the model, tokenizer for the models output of text, feature extractor for processing the
    # audio file for the input, data type for models accuracy and efficiency, processing device and to return timestamps
    # for segment of text.
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,
    )

    # Saves specified audio file path.
    audio_file_path = "./audio/Ralis.flac"

    # Performs ASR on the specified audio file with it transcribing in Lithuanian.
    result = pipe(audio_file_path, generate_kwargs={"language": "lithuanian"})
    # Saves only the text instead of chunks and text.
    transcribed_text = result["text"]

    print("Transcribed text: ", transcribed_text)

    # Ensures that text folder exists
    os.makedirs("text", exist_ok=True)
    # Saves the transcribed text to text/transcription.txt overwriting it everytime.
    with open(os.path.join("text", "transcription.txt"), "w") as text_file:
        text_file.write(transcribed_text)

    return transcribed_text

# Summarizes given transcription
def summarize():
    # Ensures that transcribed text is initialized even if the transcription.txt was not found.
    if os.path.exists("./text/transcription.txt") == True:
        with open(os.path.join("text", "transcription.txt"), "r") as text_file:
            transcribed_text = text_file.read()
        # Specifies to use llama 3.2 3B instruct by unsloth model
        model_id = "unsloth/Llama-3.2-3B-Instruct"
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch_dtype,
            device=device,
        )
        # Defines the system instruction and the users transcribed text into messages structure.
        messages = [
            {"role": "system",
             "content": "You are a summarizer bot that summarizes the given article text for informing a user of what is the most important thing mentioned in the article."},
            {"role": "user", "content": transcribed_text},
        ]
        # Performs text generation for the specified input messages.
        outputs = pipe(
            messages,
            max_new_tokens=256,
        )
        print("Summarized text: ", outputs[0]["generated_text"][-1]['content'])

        # Ensures that text folder exists
        os.makedirs("text", exist_ok=True)

        # Saves the summary text to text/summary.txt overwriting it everytime.
        with open(os.path.join("text", "summary.txt"), "w") as text_file:
            text_file.write(outputs[0]["generated_text"][-1]['content'])
    else:
        print("transcription text file was not found.")

# Results in paging file too small error
def summarizeLT():
    # Ensures that transcribed text is initialized even if the transcription.txt was not found.
    if os.path.exists("./text/transcription.txt") == True:
        with open(os.path.join("text", "transcription.txt"), "r") as text_file:
            transcribed_text = text_file.read()

        # Loads the lithuanian model tokenizer and model.
        tokenizer = AutoTokenizer.from_pretrained("neurotechnology/Lt-Llama-2-7b-instruct-hf")
        model = AutoModelForCausalLM.from_pretrained("neurotechnology/Lt-Llama-2-7b-instruct-hf")
        # Model system instructions
        PROMPT_TEMPLATE = (
            "[INST] <<SYS>> Esi paslaugus asistentas, kuris sutraukia duotą tekstą į santrauką. <</SYS>>{instruction}[/INST]"
        )

        prompt = PROMPT_TEMPLATE.format_map({'instruction': transcribed_text})
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

        # Performs text generation for the given transcription.
        outputs = model.generate(input_ids=inputs, max_new_tokens=128)
        summary = tokenizer.decode(outputs[0])
        print("Summarized text: ", summary)

        # Ensures that text folder exists.
        os.makedirs("text", exist_ok=True)

        # Saves the summary text to text/summary.txt overwriting it everytime.
        with open(os.path.join("text", "summary.txt"), "w") as text_file:
            text_file.write(summary)
    else:
        print("transcription text file was not found.")

def WER():
    if os.path.exists("./text/transcription.txt") == True & os.path.exists("./text/RalisStraipsnis.txt") == True:
        with open(os.path.join("text", "RalisStraipsnis.txt"), "r") as text_file:
            article_text = text_file.read()
        with open(os.path.join("text", "transcription.txt"), "r") as text_file:
            transcribed_text = text_file.read()
        print("Word error rate for transcription: ", jiwer.wer(article_text, transcribed_text))
    else:
        print("file was not found.")

if __name__ == '__main__':
    transcribe()
    WER()
    summarize()
    print("Built with Llama")