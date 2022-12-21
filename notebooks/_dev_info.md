- [Notes regarding data analysis](#notes-regarding-data-analysis)
  - [AVDOSVR](#avdosvr)
  - [ETL2](#etl2)
    - [Data structure](#data-structure)
    - [Feature selection](#feature-selection)
    - [Time sync](#time-sync)
    - [Preprocessing](#preprocessing)
- [`TODO`](#todo)
    - [Statistical analysis](#statistical-analysis)
    - [Feature extraction](#feature-extraction)
    - [Classification](#classification)
- [Datasets](#datasets)
- [`ETL1` - Remote Video Study - EmteqLabs](#etl1---remote-video-study---emteqlabs)
  - [Study design](#study-design)
    - [Video information in json files](#video-information-in-json-files)
  - [Dataset Notes](#dataset-notes)
- [`ETL2` Remote Video Study v2 - EmteqLabs](#etl2-remote-video-study-v2---emteqlabs)
  - [Study design](#study-design-1)
  - [Dataset Notes](#dataset-notes-1)


----

# Notes regarding data analysis

## AVDOSVR

The paper analyzes the data `AVDOSVR` originally with 43 subjects. Subjects `246` and `340` where excluded due to poor device fit. Data from participants `375` and `360` were excluded due to missing data. After preprocessing, subjects `340` and `369` were also excluded because they also have missing physiological data in `video2` containing only 56 seconds (out of 300s) and 71s of data, respectively. Subject `330` was excluded due to errors in timestamps during `video4` (see timestamps for frames 970 and 971). Therefore, the analysis includes in total the data from 37/43 participants.

## ETL2

The paper analyzes the data `ETL2` originally with 19 subjects. Data from participants `375` and `360` were excluded due to missing data. After preprocessing, subjects `340` and `369` were also excluded because they also have missing physiological data in `video2` containing only 56 seconds (out of 300s) and 71s of data, respectively. Therefore, the analysis includes in total the data from 15 participants.

### Data structure

Each participant has has `events` containing subjective data and experimental information, and `data` from the mask.
- The .json `events` was filtered to extract the time-series data for: *Valence, Arousal, RawX, RawY*.
- The .csv `data` contains outputs the following variables. Full specifications on <https://support.emteqlabs.com/data/CSV.html>:
  - `"Frame","Time","Faceplate/FaceState","Faceplate/FitState",`
  - `"HeartRate/Average","Ppg/Raw.ppg","Ppg/Raw.proximity",`
  - `"Accelerometer/Raw.x","Accelerometer/Raw.y","Accelerometer/Raw.z",`
  - `"Magnetometer/Raw.x","Magnetometer/Raw.y","Magnetometer/Raw.z",`
  - `"Gyroscope/Raw.x","Gyroscope/Raw.y","Gyroscope/Raw.z",`
  - `"Pressure/Raw"`
  - `Emg/TYPE[MUSCLE_CHANNEL]`:
      - `TYPE in [ContactStates, Contact, Raw, RawLift, Filtered, Amplitude]`
      - `MUSCLE_CHANNEL in [RightFrontalis, RightZygomaticus, RightOrbicularis, CenterCorrugator, LeftOrbicularis, LeftZygomaticus, LeftFrontalis]`

### Feature selection
- The `events` are processed as follows:
  - All the events from a participant's folder where compiled in a single file `compiled_experimental_events.csv`
  - Then, this file is processed to separate the subjective affect states (`Valence/Arousal`) to a separate file called `compiled_subjective_emotions.csv`
  - Finally, the timestamps determining the `Start` and `End` of each experimental stage are defined in the file `compiled_segment_timestamps.csv`. The information in this table can be used to filter the physiological data using the Unix timestamps for each participant and labelling it according to each stage (e.g., filtering specific video, specific Segments). Note that the code for a rest video is `VideoId = -1`.

- The initial set of `data` variables of interest are:
  - `Emg/Amplitude` from the 7 muscle channels: The Amplitude is the result of taking the raw EMG signal @ 2KHz to generate a filtered version @1KHz in the range of 100-450Hz. Movement artifacts are not removed. Then, the amplitude @50Hz is the result of applying a moving-window RMS over the filtered signal. [See here](https://support.emteqlabs.com/data/CSV.html#emgamplitude06)
  - `HeartRate/Average` sampled at 1Hz and `Ppg/Raw.ppg` @25Hz to calculate HRV
  - `Accelerometer/` @ 50Hz from the 3-axes to assess physical activity intensity.
- The **subjective emotional values** were input with the controller. `RawX` maps to `Valence` and `RawY` maps to `Arousal`.

### Time sync
- Both data sources (Events and Data) were synced via Unix timestamps.
- `Time` in the *Event files* are corrected from milliseconds `ms` using J2000 timestamps to seconds `s` in Unix timestamps, i.e., adding `+946684800000` miliseconds to the timestamp in the event.
- `Time` in *Emteq data* is originally in seconds `s` in Unix time values. Conversion to J2000 can be done using the value from `#Time/Seconds.referenceOffset`.

### Preprocessing
- The subjective emotional values were merged into the physiological data.
- All the data was resampled to periodic `50Hz` with forward fill to match the sampling highest sampling frequency of the sensor.
- The dataset for each participantt was splitted in 6 experimental **stages/segments/classes**): `[Resting_VideoPositive, VideoPositive, Resting_VideoNeutral, VideoNeutral, Resting_VideoNegative, VideoNegative]`
- For each experimental stage, the time series in `Time` were resetted to 0, each resting stage lasts `~120s` and each video stage `~300s`.

# `TODO`

### Statistical analysis
- Comparing whether the perceived emotions are actually different per participant (validating the ground-truth)


### Feature extraction
- 

### Classification
- Methods

----

# Datasets

# `ETL1` - Remote Video Study - EmteqLabs

*Contact info:* Michael Gnacek - michal.gnacek@emteqlabs.com
*Video:* https://www.youtube.com/watch?v=1snh2wMsHHM

## Study design

24 datasets were collected in a remote setting, shipping the headset with the EmteqPRO mask to the participants.

A session consists of three segments: 1) slow movement, 2) fast movement, and 3) videos. The segments 1 and 2 were designed to check how movement affects signal quality.

1. **Slow movement:** Participants followed an object while slowing moving their head in four directions: left, right,up, and down. Data quality check is at the beginning of the .json file, and data were annotated to indicate: `start`, `successful` or `failed` movement.
2. **Fast movement:** An object appears at the edge of the vision range and participants were asked to quickly move their head towards the targe, annotations were done in every action.
3. **Videos:** Participants watched videos with three categories: `positive`, `neutral`, or `negative`. Each category comprises ten 30-sec (300s in total) clips that were validated in a previous study. A 2-min relaxation video is displayed between each category. The order of the categories was randomized among participants.

### Video information in json files

- `video_1`: Data during training and the category order randomized per user

```json
{"Timestamp":677157925770,"Event":"Category sequence: Neutral, Negative, Positive"}
```

- `video_2`, `video_3`, and `video_4`: Include data for relaxation video (2-min) and videos of the selected category. It contains the annotations for:E.g.,:

```json
... pseudo-real-time valence and arousal
{"Timestamp":677157956571,"Event":"Valence:6, Arousal:2, RawX:163, RawY:69"}
... relaxation video
{"Timestamp":677157932799,"Event":"Playing rest video"}
{"Timestamp":677158052848,"Event":"Finished playing rest video"}
... start/end of video category
{"Timestamp":677158052825,"Event":"Playing category number: 1 Category name: Neutral"}
{"Timestamp":677158352899,"Event":"Video category finished"}
... start/end of video within each category
{"Timestamp":677158262900,"Event":"Playing video number: 38"}
{"Timestamp":677158292907,"Event":"Finished playing video number: 38"}
... self-ratings
{"Timestamp":677158262900,"Event":"Playing video number: 38"}
{"Timestamp":677158292907,"Event":"Finished playing video number: 38"}
```

- `video_5`: Final relaxation video before end of the study

## Dataset Notes

- A quick visual inspection for signal quality was performed on `video_4` and these participants presented good SNR in the PPG signals, useful as starting point of analysis: 101, 219, 222, 247, 248, 270, 290, 293*, 312, 314, 321*, 355*, 370, 382.
- Participants 246 and 340 could not have a suitable mask fit.
- All the datasets `_v2` have an expression calibration stage.
- Participant 375 has missing data (`video_5`) due to technical difficulties.
- Participant 360 has missing data (`slow_movement`).

# `ETL2` Remote Video Study v2 - EmteqLabs

## Study design
19 datasets were collected in a slightly different way, they are marked as `###_v2`. First, there were met physically (not remotely), but still researcher and participants were in different rooms via video call to mimick the behavior from the `v1`. 

Additionally, there was an **expression calibration stage** where the participants were asked to perform 3 repetitions of: `smile`, `frown`, `surprise`. The calibration was performed at the start of the `slow_movement_segment`. Except for participant 379, where it was accidentally left out and performed during `video_1`.

## Dataset Notes

***See the notes from `ETL1` to see the description of the json file***

