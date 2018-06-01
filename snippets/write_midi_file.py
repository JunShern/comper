import mido

mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)

# track.append(mido.Message('program_change', program=12, time=0))
track.append(mido.Message('note_on', note=64, velocity=64, time=32))
track.append(mido.Message('note_off', note=64, velocity=0, time=32))

mid.save('new_song.mid')