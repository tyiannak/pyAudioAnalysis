# <img src="icon.png" align="left" height="130"/> A Python library for audio feature extraction, classification, segmentation and applications

*This doc contains info for the test scripts used to (a) evaluate the functionality of the library (b) train the standard audio models*
 
## Train default models:
```
python3 script_train_classifiers_all.py -d "/Users/tyiannak/ResearchData/Audio Dataset/pyAudioAnalysisData/" -c svm_rbf -t sm
python3 script_train_classifiers_all.py -d "/Users/tyiannak/ResearchData/Audio Dataset/pyAudioAnalysisData/" -c svm_rbf -t movie8
python3 script_train_classifiers_all.py -d "/Users/tyiannak/ResearchData/Audio Dataset/pyAudioAnalysisData/" -c svm_rbf -t speakers
python3 script_train_classifiers_all.py -d "/Users/tyiannak/ResearchData/Audio Dataset/pyAudioAnalysisData/" -c svm_rbf -t speaker-gender
python3 script_train_classifiers_all.py -d "/Users/tyiannak/ResearchData/Audio Dataset/pyAudioAnalysisData/" -c svm_rbf -t music-genre6
python3 script_train_classifiers_all.py -d "/Users/tyiannak/ResearchData/Audio Dataset/pyAudioAnalysisData/" -c svm_rbf -t 4class

python3 script_train_classifiers_all.py -d "/Users/tyiannak/ResearchData/Audio Dataset/pyAudioAnalysisData/" -c knn -t sm
python3 script_train_classifiers_all.py -d "/Users/tyiannak/ResearchData/Audio Dataset/pyAudioAnalysisData/" -c knn -t movie8
python3 script_train_classifiers_all.py -d "/Users/tyiannak/ResearchData/Audio Dataset/pyAudioAnalysisData/" -c knn -t speakers
python3 script_train_classifiers_all.py -d "/Users/tyiannak/ResearchData/Audio Dataset/pyAudioAnalysisData/" -c knn -t speaker-gender
python3 script_train_classifiers_all.py -d "/Users/tyiannak/ResearchData/Audio Dataset/pyAudioAnalysisData/" -c knn -t music-genre6
python3 script_train_classifiers_all.py -d "/Users/tyiannak/ResearchData/Audio Dataset/pyAudioAnalysisData/" -c knn -t 4class
```

Then you have to copy the classifiers to the models folder (and to the local folder with the datasets)
```
cp models/* ../pyAudioAnalysis/data/models/
cp models/* ~/ResearchData/Audio\ Dataset/pyAudioAnalysisData/models
``` 
 
## Tests:
For testing that the current version of the library is functionable just run:

```
python3 script_tests.py 
```

or run particular shell tests:
```
sh cmd_test_00.sh ../ 
sh cmd_test_01.sh ../ 
sh cmd_test_02.sh ~/ResearchData/Audio\ Dataset/pyAudioAnalysisData/
sh cmd_test_02_B.sh ~/ResearchData/Audio\ Dataset/pyAudioAnalysisData/
sh cmd_test_02_C.sh ~/ResearchData/Audio\ Dataset/pyAudioAnalysisData/
sh cmd_test_03.sh ~/ResearchData/Audio\ Dataset/pyAudioAnalysisData/
sh cmd_test_04.sh ~/ResearchData/Audio\ Dataset/pyAudioAnalysisData/
sh cmd_test_05.sh ../
sh cmd_test_06.sh ~/ResearchData/Audio\ Dataset/pyAudioAnalysisData/
sh cmd_test_07.sh ~/ResearchData/Audio\ Dataset/pyAudioAnalysisData/
sh cmd_test_08.sh ~/ResearchData/Audio\ Dataset/pyAudioAnalysisData/
sh cmd_test_09.sh ~/ResearchData/Audio\ Dataset/pyAudioAnalysisData/
sh cmd_test_10.sh ~/ResearchData/Audio\ Dataset/pyAudioAnalysisData/
sh cmd_test_11.sh ~/ResearchData/Audio\ Dataset/pyAudioAnalysisData/ 
sh cmd_test_12_1.sh ~/ResearchData/Audio\ Dataset/pyAudioAnalysisData/
sh cmd_test_12_2.sh ~/ResearchData/Audio\ Dataset/pyAudioAnalysisData/
sh cmd_test_12_3.sh ~/ResearchData/Audio\ Dataset/pyAudioAnalysisData/
```

