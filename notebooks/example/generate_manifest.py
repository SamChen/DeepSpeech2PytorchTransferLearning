import pandas as pd
import os
import librosa

if __name__=="__main__":
    trans = pd.read_csv("transcriptions.txt")
    trans["text"] = trans["text"].str.lower()

    recording_filepath = {"uttid":[], "audio_path":[], "duration":[]}
    filenames = []
    for path, _, fs in os.walk("./audio/"):
        filenames.extend(fs)
        break
    path = os.path.abspath(path)
    for f in filenames:
        uttid = f.split('.')[0]
        audio_path = os.path.join(path, f)
        duration = librosa.get_duration(filename=audio_path)
        recording_filepath["uttid"].append(uttid)
        recording_filepath["audio_path"].append(audio_path)
        recording_filepath["duration"].append(duration)

    recording_info = pd.DataFrame(recording_filepath)

    result = pd.merge(trans, recording_info, on="uttid")

    result.to_csv("data/example.csv", index=False)
    # result.to_csv("data/val.csv")
    # result.to_csv("data/test.csv")





