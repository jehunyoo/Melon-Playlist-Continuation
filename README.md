# Melon Playlist Continuation

## Overview
[Melon Playlist Continuation](https://arena.kakao.com/c/7) is hosted by [kakao arena](https://arena.kakao.com/).<br>
Neighbor-based collaborative filtering was used.<br>
In master branch, the source codes are final version of model and our all commits & trials are in develop branch.

You can download data from [here](https://arena.kakao.com/c/7/data) only for academical purpose and make sure that you quote **kakao arena**.

For more specific ideas, please visit [my blog](https://jehunyoo.github.io/projects/melon-playlist-continuation).

### Usage
```bash
$ python inference.py
```

We assumed that data is in `res/`.

### References
- [[Paper]](https://eprints.sztaki.hu/9560/1/Kelen_1_30347064_ny.pdf) [[Code]](https://github.com/proto-n/recsys-challenge-2018) Efficient K-NN for Playlist Continuation (RecSys'18 Challenge)
- [[Paper]](https://dl.acm.org/doi/10.1145/3267471.3267481) [[Code]](https://github.com/LauraBowenHe/Recsys-Spotify-2018-challenge) Automatic Music Playlist Continuation via Neighbor-based Collaborative Filtering and Discriminative ReweightingReranking (RecSys'18 Challenge) 

### Team: dddd

#### Member of Team dddd
- [Jehun Yoo](https://github.com/JehunYoo) (percy98)
- [Daehyun Cho](https://github.com/1pha) (1pha)
- [Changgeon Lim](https://github.com/ckdrjs96) (dororo)

#### Working Repository
- [dddd](https://github.com/Arena-UOS/MelonPlaylistContinuation)
- [JehunYoo](https://github.com/Arena-UOS/JehunYoo)
- [Daehyun Cho](https://github.com/Arena-UOS/1pha)
- [Changgeon Lim](https://github.com/Arena-UOS/geon)

#### Collaboration Tools
- [Notion](https://www.notion.so/Team-dddd-ab0ca582b705420b983ad3a06c6d7e11)
- [Slack](https://kakaocompetitionuos.slack.com/)

### Score & Ranking

Leaderboard | Score | Song nDCG | Tag nDCG
---|:---:|:---:|:---:
Public | 0.327719 (10) | 0.302484 (11) | 0.470715 (18)
Final | 0.33084 (11) | 0.307122 (?) | 0.465247 (?)

&#8594; [Leaderboard](https://arena.kakao.com/c/7/leaderboard)
