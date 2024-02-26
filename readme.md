# AVDOS-VR

### Virtual Reality Affective Video Database with Physiological Signals

This repository contains scripts needed for the analysis of AVDOS-VR database. AVDOS-VR uses a subset of videos published in peer-reviewed AVDOS database.

- Data can be downloaded from [gnacek.com/affective-video-database-online-study](https://gnacek.com/affective-video-database-online-study)
- Processed data is available at the link above. It is an optional download as this file will be generated automatically when running preprocessing notebook 1
- Download the data and paste entire `data` folder into root folder generating the tree `AVDOS-VR/data/participant_XXX/*`

- To see example analyses, please refer to the corresponding publication and check the files in `AVDOS-VR/notebooks/*`
- If using root folder as the working directory, exclude the following file types from IDE search function to avoid searching through large files (*.csv, *.raw, *.json)


### Paper

- Peer-reviewed publication available at the [Nature Scientific Data](https://www.nature.com/articles/s41597-024-02953-6)

```
@article{gnacek_avdos-vr_2024,
	title = {{AVDOS}-{VR}: {Affective} {Video} {Database} with {Physiological} {Signals} and {Continuous} {Ratings} {Collected} {Remotely} in {VR}},
	author = {Gnacek, Michal and Quintero, Luis and Mavridou, Ifigeneia and Balaguer-Ballester, Emili and Kostoulas, Theodoros and Nduka, Charles and Seiss, Ellen},
	journal = {Scientific Data},
	url = {https://www.nature.com/articles/s41597-024-02953-6},
	doi = {10.1038/s41597-024-02953-6},
	shorttitle = {{AVDOS}-{VR}},
	volume = {11},
	issn = {2052-4463},
	number = {1},
}
```
