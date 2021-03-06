# face_overlay_pyto
Face overlay app sample on pyto(iOS)

# Dependency
## iOS
- iPhone(iPhone 11Pro is tested)
- Pyto(iOS app)

## PC
- Python3
- Open CV

# Setup
## iOS
- Install Pyto
- Copy `face_overlay.py` code and [karaage_icon.png](https://raw.githubusercontent.com/karaage0703/karaage_icon/master/karaage_icon.png) to your iPhone

## PC
Clone this repository.

```sh
$ git clone https://github.com/karaage0703/face_overlay_pyto/
```

Download haar-like feature for face detection and karaage_icon.

```sh
$ wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml
$ wget https://raw.githubusercontent.com/karaage0703/karaage_icon/master/karaage_icon.png
```

# Usage
## iOS
Execute code on Pyto

## PC
Execute following command:
```sh
$ python3 face_overlay.py 0
```


# License
This software is released under the MIT License, see LICENSE 

# Authors
- karaage0703

# References
