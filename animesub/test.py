class SubtitlePipeline:
    def __init__(self, context):
        self.context = context # Здесь лежат temp_dir, cancel_event, logger
        self.model_manager = ModelManager()

    def run(self, url):
        audio_path  = self.download_audio(url)
        vocals_path = self.separate_vocals(audio_path)
        timestamps  = self.detect_speech(vocals_path)
        raw_subs    = self.transcribe(timestamps)
        final_subs  = self.add_punctuation(raw_subs)
        
        self.export_srt(final_subs)