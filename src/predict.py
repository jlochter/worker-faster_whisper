"""
This file contains the Predictor class, which is used to run predictions on the
Whisper model. It is based on the Predictor class from the original Whisper
repository, with some modifications to make it work with the RP platform.
"""

import gc
import threading

from runpod.serverless.utils import rp_cuda

from faster_whisper import WhisperModel

# Define available models (for validation)
AVAILABLE_MODELS = {
    "large-v2",
    "large-v3",
    "distil-large-v2",
    "distil-large-v3",
}


class Predictor:
    """A Predictor class for the Whisper model with lazy loading"""

    def __init__(self):
        """Initializes the predictor with no models loaded."""
        self.models = {}
        self.model_lock = threading.Lock()

    def setup(self):
        """No models are pre-loaded. Setup is minimal."""
        pass

    def predict(
        self,
        audio,
        model_name="large-v2",
        language="ja",
        chunk_length=30,
    ):
        """
        Run a single prediction on the model, loading/unloading models as needed.
        """
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model name: {model_name}. Available models are: {AVAILABLE_MODELS}"
            )

        with self.model_lock:
            model = None
            if model_name not in self.models:
                # Unload existing model if necessary
                if self.models:
                    existing_model_name = list(self.models.keys())[0]
                    print(f"Unloading model: {existing_model_name}...")
                    del self.models[existing_model_name]
                    self.models.clear()
                    gc.collect()
                    print(f"Model {existing_model_name} unloaded.")

                # Load the requested model
                print(f"Loading model: {model_name}...")
                try:
                    loaded_model = WhisperModel(
                        model_name,
                        device="cuda" if rp_cuda.is_available() else "cpu",
                        compute_type="float16" if rp_cuda.is_available() else "int8",
                    )
                    self.models[model_name] = loaded_model
                    model = loaded_model
                    print(f"Model {model_name} loaded successfully.")
                except Exception as e:
                    print(f"Error loading model {model_name}: {e}")
                    raise ValueError(f"Failed to load model {model_name}: {e}") from e
            else:
                model = self.models[model_name]
                print(f"Using already loaded model: {model_name}")

            if model is None:
                raise RuntimeError(
                    f"Model {model_name} could not be loaded or retrieved."
                )

        segments, info = model.transcribe(
            str(audio),
            language=language,
            word_timestamps=True,
            chunk_length=chunk_length,
            log_progress=True,
        )

        segments = list(segments)

        word_timestamps_list = []
        for segment in segments:
            for word in segment.words:
                word_timestamps_list.append(
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                    }
                )

        return {
            "segments": serialize_segments(segments),
            "detected_language": info.language,
            "word_timestamps": word_timestamps_list,
            "device": "cuda" if rp_cuda.is_available() else "cpu",
            "model": model_name,
        }


def serialize_segments(transcript):
    """
    Serialize the segments to be returned in the API response.
    """
    return [
        {
            "id": segment.id,
            "seek": segment.seek,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "tokens": segment.tokens,
            "temperature": segment.temperature,
            "avg_logprob": segment.avg_logprob,
            "compression_ratio": segment.compression_ratio,
            "no_speech_prob": segment.no_speech_prob,
        }
        for segment in transcript
    ]


