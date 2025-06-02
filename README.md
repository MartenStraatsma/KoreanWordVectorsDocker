# Korean Word Vectors Docker
A fork of [KoreanWordVectors](https://github.com/SungjoonPark/KoreanWordVectors) by Park SungJoon.
Updated to the most recent release of [fastText](https://github.com/facebookresearch/fastText/).
For questions about the fastText implementation, refer to these.

## Training Data
Run [data.sh](/src/util/data.sh) (with the optional argument `train`, `dev`, or `test`) to download and parse a 2020 corpus based on a NamuWiki crawl.

All data needs to be deconstructed into separate 초성, 중성, and, 종성.
To this end, make sure all data is parsed with [decompose_letters.py](/src/util/decompose_letters.py):
```bash
python3 decompose_letters.py [input file] [output file]
```

## Docker
To build an image, change to the [src](/src/) directory and execute
```bash
docker build . -t fasttextdocker
```

To run the container, execute
```bash
docker run --mount type=bind,source="$(pwd)",target=/var/www -d fasttextdocker
```