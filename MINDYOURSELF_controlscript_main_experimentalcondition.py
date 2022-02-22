#%% IMPORTS
from pylsl import StreamInlet, resolve_stream
import rtmidi
import time
import pandas as pd
import numpy as np
import random
import serial
from datetime import datetime
#%% LOAD IN DATA of music samples and corresponding MIDI notes and corresponding val and ar
df = pd.read_csv("C:/Users/s2365332/Downloads/VAL_AR.csv")
df = pd.read_csv("C:/Users/s2365332/Downloads/VAL_AR.csv")

df_note = df.iloc[0:60, 3].fillna(0).astype(int)
df_val = df.iloc[0:60, 13].fillna(0).astype(int)
df_ar = df.iloc[0:60, 14].fillna(0).astype(int)

df_note = np.reshape(np.array(df_note), (15, 4))
df_val = np.reshape(np.array(df_val), (15, 4))
df_ar = np.reshape(np.array(df_ar), (15, 4))
#%% FUNCTION DEFINITIONS


def max_min_int(value):  # Taking out extreme values and rounding the features
    if value < -9:
        value = -9
    elif value > 10:
        value = 10
    else:
        value = int(value)
    return value


def lower_arousal(list_of_indices, error):
    for i in range(abs(error)):
        if not(sum(list_of_indices[:3]) == 0):
            action = 'percussion'
        else:
            action = 'melody'
        if action == 'percussion':
            options = np.nonzero(list_of_indices[:3])[0]
            if len(options) > 0:
                index = np.random.choice(options)
                list_of_indices[index] -= 1
            else:
                list_of_indices = list_of_indices

        elif action == 'melody':
            options = np.nonzero(list_of_indices[3:])[0] + 3
            if len(options) > 0:
                index = np.random.choice(options)
                list_of_indices[index] = 0
            else:
                list_of_indices = list_of_indices
    return list_of_indices


def raise_arousal(list_of_indices, error):
    for i in range(error):
        if sum(list_of_indices[:3]) == 9:
            action = 'melody'
        elif len(np.nonzero(list_of_indices[3:])[0]) == 12:
            action = 'percussion'
        else:
            action = random.choice(['percussion', 'melody'])
        if action == 'percussion':
            options = np.where(list_of_indices[:3] < 3)[0]
            if len(options) > 0:
                index = np.random.choice(options)
                list_of_indices[index] += 1
            else:
                list_of_indices = list_of_indices

        elif action == 'melody':
            options = np.where(list_of_indices[3:] == 0)[0] + 3
            if len(options) > 0:
                index = np.random.choice(options)
                list_of_indices[index] = random.choice([1, 2, 3])
            else:
                list_of_indices = list_of_indices
    return list_of_indices


def change_arousal(list_of_indices, goal_arousal):
    current_arousal = int(track_arousal_weight * (np.count_nonzero(list_of_indices[3:]) + np.sum(list_of_indices[:3])))
    clipped_goal_arousal = np.clip(np.array([goal_arousal]), -9, 10)[0]
    error = (clipped_goal_arousal + 10) - current_arousal

    if error > 0:
        ar_updated_list_of_indices = raise_arousal(list_of_indices, error)

    elif error < 0:
        ar_updated_list_of_indices = lower_arousal(list_of_indices, error)

    else:
        ar_updated_list_of_indices = list_of_indices

    return ar_updated_list_of_indices


def lower_valence(list_of_indices, error):
    for i in range(abs(error)):
        options = np.where(list_of_indices[3:] > 1)[0] + 3
        if len(options)> 0:
            index = np.random.choice(options)
            list_of_indices[index] -= 1
        else:
            list_of_indices = list_of_indices
    return list_of_indices


def raise_valence(list_of_indices, error):
    for i in range(error):
        low_enough = list_of_indices[3:] < 3
        high_enough = list_of_indices[3:] > 0
        if (len(low_enough) > 0) and (len(high_enough) > 0):
            options = np.where(high_enough & low_enough)[0] + 3
            if len(options) > 0:
                index = np.random.choice(options)
                list_of_indices[index] += 1
            else:
                list_of_indices = list_of_indices
        else:
            list_of_indices = list_of_indices
    return list_of_indices


def change_valence(list_of_indices, goal_valence):
    clipped_goal_valence = np.clip(np.array([goal_valence]), -10, 10)[0]
    valence_on = list_of_indices[np.nonzero(list_of_indices[3:])[0] + 3]
    if len(valence_on) > 0:
        current_valence = np.mean((valence_on - 2) * 10)
        error = clipped_goal_valence - current_valence
        repetitions = int(error * 1.2)

        if error > 0:
            val_updated_list_of_indices = raise_valence(list_of_indices, repetitions)

        elif error < 0:
            val_updated_list_of_indices = lower_valence(list_of_indices, repetitions)

        else:
            val_updated_list_of_indices = list_of_indices
    else:
        val_updated_list_of_indices = list_of_indices

    return val_updated_list_of_indices


def indices_to_notes(list_of_indices):
    list_of_notes = np.zeros(15)

    for i, j in enumerate(list_of_indices):
        list_of_notes[i] = df_note[i][j]

    return list_of_notes


def update_list_of_indices(list_of_indices, list_of_changes):
    updated_list_of_indices = list_of_indices

    for note in list_of_changes:
        location, number = np.where(df_note == note)
        updated_list_of_indices[location - 1] = number

    return updated_list_of_indices


def update_notes(list_of_indices, goal_arousal, goal_valence):
    old_list_of_notes = indices_to_notes(list_of_indices)
    ar_updated_list_of_indices = change_arousal(list_of_indices, goal_arousal)
    val_updated_list_of_indices = change_valence(ar_updated_list_of_indices, goal_valence)
    new_list_of_notes = indices_to_notes(val_updated_list_of_indices)
    change_notes = new_list_of_notes[~(new_list_of_notes == old_list_of_notes)]
    if len(change_notes) > max_changes:
        downsized_list_of_changes = np.random.choice(change_notes, max_changes, replace=False)
    else:
        downsized_list_of_changes = change_notes

    return downsized_list_of_changes


def fetch_port_number():
    midi_out = rtmidi.MidiOut()
    available_ports = midi_out.get_ports()
    print(available_ports)
    del midi_out
    return


def send_midi(list_of_notes):
    for note in list_of_notes:
        midi_out = rtmidi.MidiOut()
        available_ports = midi_out.get_ports()

        if available_ports:
            midi_out.open_port(midiport)
        else:
            midi_out.open_virtual_port("My virtual output")

        with midi_out:
            note_on = [0x90, note, 112] # channel 1, middle C, velocity 112
            note_off = [0x80, note, 0]
            midi_out.send_message(note_on)
            time.sleep(0.5)
            midi_out.send_message(note_off)
            time.sleep(0.1)

        del midi_out
    return


def send_cc_midi(list_of_values):
    for i, value in enumerate(list_of_values):
        midi_out = rtmidi.MidiOut()
        available_ports = midi_out.get_ports()

        if available_ports:
            midi_out.open_port(midiport)
        else:
            midi_out.open_virtual_port("My virtual output")

        with midi_out:
            # MIDI CC = [.. , tempo, volume, reverb, underwater]
            cc = [0xB0, (i + 1), value]
            midi_out.send_message(cc)
            time.sleep(0.5)

        del midi_out
    return


def update_cc(old_cc, goal_cc):
    new_cc = np.array([0, 0, 0, 0, 0])
    for i, value in enumerate(old_cc[0]):
        error = np.clip((goal_cc[i] - value), -max_midi_change, max_midi_change)
        new_cc[i] = int(value + error)

    return new_cc


def update_rgb(old_rgb, goal_rgb):
    new_rgb = np.array([0, 0, 0, 0])



    for i, value in enumerate(old_rgb[0]):
        error = np.clip((goal_rgb[i] - value), -max_light_change, max_light_change)
        new_rgb[i] = int(value + error)

    lowest = np.amin(new_rgb[0:3])
    lowest_loc = np.where(new_rgb == lowest)
    new_rgb[lowest_loc] = lowest/extremizer

    highest = np.amax(new_rgb[0:3])
    highest_loc = np.where(new_rgb == highest)
    new_rgb[highest_loc] = highest * extremizer

    new_rgb = np.clip(new_rgb, 0, 255)
    return new_rgb


def change_LEDs(rgb):
    red, green, blue, wait = rgb
    message = str(red) + ":" + str(green) + ":" + str(blue) + ":" + str(wait)
    serialcomm.write(message.encode())
    print("changing LEDs to: " + message)
    return



#%% MAIN UPDATE LOOP


def main():
    data = dict()
    data['info'] = ['ar', 'val', 'fru', 'eng', 'val', 'exc', 'foc', 'med']

    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    stream_arval = resolve_stream('name', 'Arousal-Valence')
    stream_perfmet = resolve_stream('name', 'EmotivDataStream-Performance-Metrics')

    # create a new inlet to read from the stream
    inlet_arval = StreamInlet(stream_arval[0])
    inlet_perfmet = StreamInlet(stream_perfmet[0])

    # some defaults to start with
    list_of_indices = np.array([0, 0, 0, 1, 2, 0, 0, 1, 0, 3, 0, 0, 2, 3, 0])
    ar, valence = [0, 0]
    fru, eng, val, exc, foc, med = [0.2, 0.5, 0.5, 0.5, 0.5, 0.5]

    initial_soundscape = indices_to_notes(list_of_indices)
    send_midi(initial_soundscape)

    new_rgb = np.array([100, 100, 100, 100])
    new_cc_values = np.array([50, 50, 50, 50, 50])

    try:
        while True:
            # get a new sample (you can also omit the timestamp part if you're not
            # interested in it)
            chunk_arval, timestamps_arval = inlet_arval.pull_chunk()
            chunk_perfmet, timestamps_perfmet = inlet_perfmet.pull_chunk()

            if timestamps_arval:
                ar = max_min_int(chunk_arval[0][0])  # arousal extracted via OpenVibe
                valence = max_min_int(chunk_arval[0][1])  # valence extracted via OpenVibe
                print("arousal: " + str(ar) + ", valence: " + str(valence))
                old_list_of_indices = np.array(list_of_indices)
                list_of_changes = update_notes(list_of_indices, ar, valence)

                send_midi(list_of_changes)
                list_of_indices = update_list_of_indices(old_list_of_indices, list_of_changes)

            if timestamps_perfmet:
                old_rgb = np.array([new_rgb])
                old_cc_values = np.array([new_cc_values])

                perfmet = np.array(chunk_perfmet[0][1:])
                print(perfmet)
                if (perfmet < 0).any():
                    perfmet = np.array([0.7, 0.3, 0.3, 0.4, 0.6, 0.3])
                    perfmet2 = np.clip(perfmet - (np.random.rand(6) * 2 - 1) * 0.25, 0, 1)
                    print("RANDOMISED PERFORMANCE METRICS")

                    now = datetime.now()
                    data[now] = "NaN" # if there is a problem with the EEG, the data is not printed into the file.
                    # participants with a lot of missing data, can be taken out of the set.

                else:
                    perfmet2 = perfmet

                    # save data only if we have enough contact with EEG to create a perfomance metrics
                    now = datetime.now()
                    data[now] = [ar, valence, perfmet2]

                fru, eng, val, exc, foc, med = perfmet2


                print("fru: " + str(fru) + ", val: " + str(val) + ", exc: " + str(exc)
                      + ", foc: " + str(foc) + ", med: " + str(med))

                goal_rgb = np.array([fru * 0.8, foc, med, (1-fru)]) * 255
                goal_cc = np.array([fru, exc, (1-fru), foc, med]) * 127

                new_rgb = update_rgb(old_rgb, goal_rgb)
                new_cc_values = update_cc(old_cc_values, goal_cc)

                change_LEDs(new_rgb)
                send_cc_midi(new_cc_values)

    except KeyboardInterrupt:
        #When the program is stopped, the data is written in a CSV file.
        df = pd.DataFrame.from_dict(data, orient='index')
        df.to_csv(('participant' + str(participant_number) +'.csv'))

#%% reset serial port
serialcomm = serial.Serial('COM4', 9600)
serialcomm.close()


#%% CONSTANTS
fetch_port_number()  # check if the right MIDI port indeed still is 2 (LoopMIDI Port)
midiport = 1

max_changes = 7  # determine the maximum number of changing samples per time step, for the sake of continuity
track_arousal_weight = 1.5 # the weight of the tracks to calculate arousal.

max_light_change = 200
max_midi_change = 60
# If higher, less tracks will be activated simultaneously.

extremizer = 1.5

serialcomm = serial.Serial('COM4', 9600)
serialcomm.timeout = 1

participant_number = 0



#%% RUN THE WHOLE SHABANG

if __name__ == '__main__':
    main()


#%%

