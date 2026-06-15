import os
import warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'        
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'         
warnings.filterwarnings("ignore")

import numpy as np
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

notes = []

print("--- PARSING MIDI FILES ---")
for file in os.listdir("midi_songs"):
    if file.endswith(".mid") or file.endswith(".midi"):
        try:
            midi = converter.parse(os.path.join("midi_songs", file))
            print(f"Successfully parsed: {file}")
            
            try:
                parts = instrument.partitionByInstrument(midi)
                if parts and len(parts.parts) > 0:
                    # Look through all instrument layers
                    notes_to_parse = midi.recurse() 
                else:
                    notes_to_parse = midi.flat.recurse()
            except:
                notes_to_parse = midi.flat.recurse()
                
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Skipping damaged file {file} due to error: {e}")

print(f"\nTotal note elements extracted: {len(notes)}")

print("\n--- PREPROCESSING SEQUENCES ---")
sequence_length = 50
pitches = sorted(list(set(notes)))
vocab_size = len(pitches)
note_to_int = {note: num for num, note in enumerate(pitches)}

network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)
X = np.reshape(network_input, (n_patterns, sequence_length, 1))
X = X / float(vocab_size)
y = to_categorical(network_output)

print("\n--- INITIALIZING LSTM ARCHITECTURE ---")
model = Sequential([
    LSTM(128, input_shape=(X.shape[1], X.shape[2])), 
    Dropout(0.2),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

print("\n--- TRAINING DEEP LEARNING MODEL ---")
model.fit(X, y, epochs=25, batch_size=256)

print("\n--- GENERATING NEW MUSIC PATTERNS ---")
start_idx = np.random.randint(0, len(network_input) - 1)
int_to_note = {num: note for num, note in enumerate(pitches)}
pattern = network_input[start_idx]
prediction_output = []

temperature = 0.8 
for note_index in range(200):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(vocab_size)
    prediction = model.predict(prediction_input, verbose=0)[0]
    prediction = np.log(prediction + 1e-7) / temperature
    exp_preds = np.exp(prediction)
    prediction = exp_preds / np.sum(exp_preds)
    index = np.random.choice(len(prediction), p=prediction)
    result = int_to_note[index]
    prediction_output.append(result)
    pattern.append(index)
    pattern = pattern[1:]

print("\n--- EXPORTING GENERATED SEQUENCE TO MIDI ---")
offset = 0
output_notes = []

for pattern in prediction_output:
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        chord_notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            chord_notes.append(new_note)
        new_chord = chord.Chord(chord_notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)
    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='generated_output.mid')
print("\n[SUCCESS] Music generation complete! Saved file as: generated_output.mid")