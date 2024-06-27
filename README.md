# ![facelog thumbnail][facelog-thumbnail] FaceLog ![facelog badge][facelog-badge]
A webpage application that can record attendance using the attendee's face. This application provides essential tools to manage attendance during runtime conveniently. Registering new attendees, training models and checking attendance can be done through the web app directly.

## Table of Contents
- [Features](#features)
- [Guidelines](#guidelines)
- [Installation](#installation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features
### Main Features
![facial recognition][facial-recognition] &nbsp;
![face apturing][face-capturing]
- **Start Facial Recognition** - Start the real-time facial recognition to record the attendance.
- **Register User** - Capture images of the attendee's face for the system to recognize the face.

![chart][chart] &nbsp;
![attendance table][attendance-table]
- **Update Model** - Train the model using the captured faces dataset. This can take some time depending on the size of the dataset.
- **Analyze Model** - Generate a scatter plot chart in a 2D plane representing the current model data distribution.
- **View Attendance** - Opens the attendance table and displays the existing attendance from the Excel files.

### Additional Features
![password change][password-change] &nbsp;
![camera selector][camera-selector] &nbsp;
![attendance selector][attendance-selector]
- **Change Password** - Change the system's password to improve security.
- **Camera Selection** - Switch between available cameras connected to the device.
- **Attendance File Selection** - Switch between available attendance files.
- **Button Tooltips** - Helps user understand the button's functionality.
- **Dynamic Buttons** - Helps user identify the button and system state.
- **Protected Buttons** - Prevent sensitive actions from being executed without permission.

> [!IMPORTANT]
> The default password is `admin`.

> [!IMPORTANT]
> The system automatically deletes datasets with no face or multiple faces, but sometimes, patterns in the background can be recognized as faces and the program cannot differentiate the error. It requires manual filtering which is why `facefilter.py` is created for that purpose.

> [!NOTE]
> The system performs facial recognition every 0.1 seconds to improve performance. It will recognize face at 30% probability but only record attendance if above 70% probability.

> [!WARNING]
> Always have a backup of your dataset since training the model can delete invalid datasets.

## Guidelines
### Dataset Capturing Guidelines
1. Keep the background plain without patterns, movements and other people.
2. It is recommended to take the face dataset capturing two times. One in a well-lit environment and another in a darker environment.
3. During capturing, try to move the attendees' heads at different angles slowly. Move from left to right and try to write the letter `O` using their head.
4. During capturing, try to mimic different common emotions, visible teeth and talking.
5. Capturing 50 face variations is not required but it would be ideal as long as the attendees are still comfortable during the process. At least providing 30 images can be acceptable but can produce less desirable results.

### Dataset Management Guidelines
1. Datasets are stored in the `faces` folder. Each attendee's name is inside the folder and the system uses the folder name as the attendee's name.
2. In case of capturing a dataset of the same attendees where there is a name conflict, you can temporarily add numbers to the attendee's name during registration and then move the images to the proper folder later.
2. In case of gathering datasets from external sources like social media and personal photos, you can move those images directly to the proper attendee's folder assuming that this image was a valid dataset based on [Dataset Capturing Guidelines](#dataset-capturing-guidelines)

## Installation
1. Download and install the latest version of [Python][python].
2. Install the executable file. Make sure to include `pip` in the installation.
3. Go to the latest [release page][release-page] of FaceLog.
4. Download and extract `Facelog-0.1.0-Beta.zip`.
5. Open the terminal inside the folder and `run pip install -r requirements.txt`
6. Wait for the installation of nine dependencies.
7. Once all dependencies are installed, type `python app.py` in the terminal. This will start the flask server.
8. Navigate to `https://localhost:5000` using your preferred browser to access the web application.

## License
This project is licensed under the Unlicense License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **[Visual Studio Code][visual-studio-code]**: For development environment.
- **[css.gg][css-gg]**: For icons.
- **[opencv-python][opencv-python]**: For image processing.
- **[numpy][numpy]**: For handling different types of arrays.
- **[scikit-learn][scikit-learn]**: For machine learning and statistical modeling.
- **[flask_cors][flask-cors]**: For handling resource sharing between Python and JavaScript.
- **[facenet-pytorch][facenet-pytorch]**: For providing a pre-trained model for facial recognition.
- **[pandas][pandas]**: For data manipulation and analysis of Excel data.
- **[openpyxl][openpyxl]**: For reading and writing Excel files.
- **[matplotlib][matplotlib]**: For generating scatter-plot chart.
- **[cryptography][cryptography]**: For generating the key, encrypting and decrypting.
- **[LSPU-SPCC BSCS2A A.Y 2023-2024][lspu-spcc-bscs2a-ay-2023-2024]**: For contributing their faces dataset.

<!-- Reference -->
[facelog-thumbnail]: https://github.com/Mindkerchief/FaceLog/assets/130748576/0e2ea03f-d343-4ba2-9807-f507ac6cfe3d
[facelog-badge]: https://img.shields.io/badge/WebApp-Real_time_Facial_Recognition_Attendance_System-6850A8

[facial-recognition]: https://github.com/Mindkerchief/FaceLog/assets/130748576/606640e0-acb8-41d1-8a3e-0dcda1857c30
[face-capturing]: https://github.com/Mindkerchief/FaceLog/assets/130748576/f9ac633e-32eb-4fda-91ff-2f9b12edf2db
[chart]: https://github.com/Mindkerchief/FaceLog/assets/130748576/02271d0b-b6ed-4a82-8cea-33832139055b
[attendance-table]: https://github.com/Mindkerchief/FaceLog/assets/130748576/5e7db092-882c-4ed5-8fc4-65eda360a82b
[password-change]: https://github.com/Mindkerchief/FaceLog/assets/130748576/e0f79d1c-c84c-4268-b7f7-1c5c5ff7248c
[camera-selector]: https://github.com/Mindkerchief/FaceLog/assets/130748576/54b4918b-64fb-4ddb-832d-23098450c6df
[attendance-selector]: https://github.com/Mindkerchief/FaceLog/assets/130748576/debc2548-9ce1-40b0-bfdb-b77e3800c627

[release-page]: https://github.com/Mindkerchief/FaceLog/releases
[python]: https://www.python.org/downloads/
[visual-studio-code]: https://code.visualstudio.com/docs
[css-gg]: https://css.gg/
[opencv-python]: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
[numpy]: https://numpy.org/doc/stable/index.html
[scikit-learn]: https://scikit-learn.org/0.21/documentation.html
[flask-cors]: https://flask-cors.readthedocs.io/en/latest/api.html
[facenet-pytorch]: https://github.com/timesler/facenet-pytorch
[pandas]: https://pandas.pydata.org/docs/
[openpyxl]: https://openpyxl.readthedocs.io/en/stable/
[matplotlib]: https://matplotlib.org/stable/users/index
[cryptography]: https://cryptography.io/en/latest/
[lspu-spcc-bscs2a-ay-2023-2024]: https://web.facebook.com/photo.php?fbid=626282756192961&set=a.626292752858628&type=3
