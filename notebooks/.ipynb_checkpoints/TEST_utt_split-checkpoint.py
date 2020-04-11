#!/usr/bin/env python

import gentle
import pandas as pd
from collections import defaultdict
import copy

class utterances_split():
    def __init__(self):
        self.resources = gentle.Resources()
        
        
    def trans_alignment(self, audiofile, transcript, st, duration, nthreads=1, disfluency=False, conservative=False, disfluencies = set(['uh', 'um'])):
        # resampled will chuck the audio file on fly. And return an object of the chuncked audio
        try:
            with gentle.resampled(audiofile, offset=st, duration=duration) as wavfile:
                aligner = gentle.ForcedAligner(self.resources, transcript, nthreads=nthreads, disfluency=disfluency, conservative=conservative, disfluencies=disfluencies)
                result = aligner.transcribe(wavfile)
            return result.words
        except:
            print("something wrong with {}.\n st:{}, duration:{}".format(audiofile, st, duration))
            return []


    def audio_segmentation(self, align: list, global_st: float, max_duration: float, max_gap: float):
        i = 0
        length = len(align)
        segs = []
        while i < length:
            seg, i = self.words_accumulate(align, i, max_duration, max_gap)
            if seg:
                segs.append((i, seg))
        return segs
    

    def words_accumulate(self, align, st_position, max_duration, max_gap):
        words = []
        # reture case 1: invalide st position
        if align[st_position].case != "success":
            return None, st_position+1

        local_st = align[st_position].start
        prev_et = align[st_position].end
        duration = 0
        for index, word in enumerate(align[st_position:], st_position):
            # return case 2: get invalide word
            if word.case != "success":
                return words, index+1

            # reture case 3:
            duration = word.end - local_st
            gap = word.start - prev_et
            if (duration > max_duration) or (gap > max_gap):
                return words, index+1

            # operation
            words.append(word)
            prev_et = word.end

        # return case 4: end of alignment
        return words, index+1


    def info_extraction(self, seg, global_st):
        # given a list of Word objects,
        st = seg[0].start + global_st
        et = seg[-1].end + global_st
        duration = et-st
        words = " ".join([i.word for i in seg])
        return words, st, et, duration

    def utt_split(self, row, max_duration=7.0, max_gap=1.0, disfluencies=set(['uh', 'um'])):
        transcript = row.text
        audiofile = row.audio_path
        global_st = row.st
        duration = row.duration

        new_utts = []
        align = self.trans_alignment(audiofile=audiofile,
                                transcript=transcript,
                                st = global_st,
                                duration=duration,
                                disfluencies = disfluencies)
        segs = self.audio_segmentation(align=align,
                                  global_st=global_st,
                                  max_duration=max_duration,
                                  max_gap=max_gap)

        for index, seg in segs:

            utt, st, et, duration = self.info_extraction(seg, global_st)
            temp_utt = copy.deepcopy(row)
            temp_utt.uttid = "{}_{}".format(temp_utt.uttid, index)
            temp_utt.duration = duration
            temp_utt.text = utt
            temp_utt.st = st
            temp_utt.et = et
            temp_utt.audio_path = audiofile

            new_utts.append(temp_utt)

        return new_utts
