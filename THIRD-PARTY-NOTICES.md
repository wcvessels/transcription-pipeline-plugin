# Third-party notices

`transcription-pipeline-plugin` is MIT-licensed (see `LICENSE`). It runs on, and depends on, third-party software and machine-learning model weights that carry their own licenses.

This repository ships source code and sha256-pinned download URLs only. It does NOT bundle or redistribute any model weights. The diarization weights are auto-fetched at first run from ungated public mirrors, sha256-verified against the official files, and cached locally. The notices below are provided as attribution and as good practice for the models and toolkits this tool downloads and uses.

## Speaker diarization model weights (pyannote)

The diarization pipeline (`pyannote/speaker-diarization-3.1`) loads two weight files. They do not share one license.

### Segmentation weights: pyannote/segmentation-3.0

Licensed MIT. The copyright year on the weights notice is 2023 (the pyannote-audio code repository uses 2020; this notice is for the weights).

```
MIT License

Copyright (c) 2023 CNRS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Embedding weights: pyannote/wespeaker-voxceleb-resnet34-LM

Licensed CC-BY-4.0.

```
Speaker-embedding model: pyannote/wespeaker-voxceleb-resnet34-LM
(c) Hervé Bredin / pyannote. A PyTorch wrapper around WeSpeaker's
voxceleb_resnet34_LM model.
Licensed under the Creative Commons Attribution 4.0 International License
(CC-BY-4.0): https://creativecommons.org/licenses/by/4.0/
The weights are a derived/adapted artifact trained by the WeSpeaker project
(wenet-e2e, Apache-2.0 toolkit code) on the VoxCeleb dataset (KAIST; CC-BY-4.0).
Changes: redistributed unmodified (auto-fetched at runtime, not bundled) as part
of transcription-pipeline-plugin; no modification to the weights.
```

## Supporting toolkits and code

Referenced by the pipeline but not bundled or redistributed by this repository.

- pyannote-audio (CNRS): MIT, Copyright (c) 2020 CNRS. https://github.com/pyannote/pyannote-audio
- WeSpeaker (wenet-e2e): Apache-2.0. https://github.com/wenet-e2e/wespeaker

## Python dependencies

WhisperX, faster-whisper, PyTorch, and the other packages listed in `requirements.txt`, plus ffmpeg, are installed from their official distributions. Each carries its own upstream license. This project bundles none of them.

## Academic citations

The pyannote and WeSpeaker maintainers request the following citations.

```bibtex
@inproceedings{Plaquet23,
  author={Alexis Plaquet and Hervé Bredin},
  title={{Powerset multi-class cross entropy loss for neural speaker diarization}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}

@inproceedings{Bredin23,
  author={Hervé Bredin},
  title={{pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}

@inproceedings{Wang2023,
  title={Wespeaker: A research and production oriented speaker embedding learning toolkit},
  author={Wang, Hongji and Liang, Chengdong and Wang, Shuai and Chen, Zhengyang and Zhang, Binbin and Xiang, Xu and Deng, Yanlei and Qian, Yanmin},
  booktitle={ICASSP 2023, IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

## VoxCeleb dataset

The embedding model's provenance chain includes the VoxCeleb dataset.

- Nagrani, A., Chung, J.S., Zisserman, A. "VoxCeleb: a large-scale speaker identification dataset," Interspeech 2017.
- Chung, J.S., Nagrani, A., Zisserman, A. "VoxCeleb2: Deep Speaker Recognition," Interspeech 2018.
